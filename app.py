import sys
try:
    # Render's default sqlite3 version is too old for ChromaDB
    # This hot-swaps the system sqlite3 with the newer pysqlite3-binary package
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a clean, responsive UI
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stChatMessage"] {
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .st-emotion-cache-1c7y2kd {
        flex-direction: row-reverse;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_models():
    """Lazily initialize models and cache them for Streamlit."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    system_prompt = (
        "You are an expert assistant. Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you don't know.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return embeddings, llm, question_answer_chain

def process_pdfs(uploaded_files):
    """Process uploaded PDFs and build the RAG chain."""
    if not uploaded_files:
        return None
        
    all_docs = []
    
    # Use a temporary directory to handle file loading via PyPDFLoader
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            loader = PyPDFLoader(temp_path)
            all_docs.extend(loader.load())
            
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(all_docs)
    
    embeddings, _, qa_chain = get_models()
    
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    rag_chain = create_retrieval_chain(retriever, qa_chain)
    return rag_chain


# --- UI LAYOUT ---

st.title("📚 Intelligent Document Assistant")
st.markdown("Upload your PDF documents and ask questions about their content right away!")

# Sidebar setup for file uploads
with st.sidebar:
    st.header("📂 Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF files", 
        type=["pdf"], 
        accept_multiple_files=True,
        help="Select one or more PDF files to analyze."
    )
    
    if st.button("Process Documents", type="primary", use_container_width=True):
        if uploaded_files:
            with st.spinner("Processing & embedding documents..."):
                st.session_state.rag_chain = process_pdfs(uploaded_files)
                st.session_state.chat_history = []  # Reset chat on new uploads
                st.success(f"Successfully processed {len(uploaded_files)} document(s)!")
        else:
            st.error("Please upload at least one PDF file.")
            
    st.divider()
    st.markdown("### Powered By")
    st.markdown("- **Google Gemini 2.5 Flash**")
    st.markdown("- **LangChain**")
    st.markdown("- **ChromaDB**")
    st.markdown("- **Sentence Transformers**")

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

st.divider()

# Display Chat History
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input & Interaction
if query := st.chat_input("Ask a question about your documents..."):
    if st.session_state.rag_chain is None:
        st.error("⚠️ Please upload and process at least one document in the sidebar first.")
    else:
        # User message
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        # Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                try:
                    response = st.session_state.rag_chain.invoke({"input": query})
                    answer = response["answer"]
                    st.markdown(answer)
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
