import os
import shutil
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
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

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Directory to temporarily store uploaded PDFs
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Global variables for our ML models
rag_chain = None

print("Initializing embedding model and LLM (this happens once)...")
# Initialize embeddings and LLM upfront so we don't have to wait every time someone uploads
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


@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    """Serve the basic HTML frontend page on a GET request."""
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={"query": None, "answer": None, "system_status": None}
    )

@app.post("/upload", response_class=HTMLResponse)
async def handle_upload(request: Request, files: list[UploadFile] = File(...)):
    """Handle PDF uploads, process them, and re-create the RAG chain."""
    global rag_chain
    
    saved_files = []
    # Save the uploaded files to disk
    for file in files:
        if file.filename.endswith(".pdf"):
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_files.append(file_path)
            
    if not saved_files:
        return templates.TemplateResponse(
            request=request, 
            name="index.html", 
            context={"query": None, "answer": None, "system_status": "No valid PDF files uploaded."}
        )
        
    print(f"Processing {len(saved_files)} file(s)...")
    
    # Process all uploaded PDFs
    all_docs = []
    for file_path in saved_files:
        loader = PyPDFLoader(file_path)
        all_docs.extend(loader.load())
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(all_docs)
    
    # Update Vector Store
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Create the new RAG chain for this context
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    status_msg = f"Successfully processed {len(saved_files)} document(s)! You can now ask questions."
    
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={"query": None, "answer": None, "system_status": status_msg}
    )

@app.post("/", response_class=HTMLResponse)
async def handle_form(request: Request, query: str = Form(...)):
    """Handle the user's form submission and return the LLM answer."""
    global rag_chain
    
    if rag_chain is None:
         return templates.TemplateResponse(
             request=request, 
             name="index.html", 
             context={
                "query": query, 
                "answer": "No documents uploaded yet. Please upload PDF(s) first.",
                "system_status": None
            }
         )

    # Invoke the dynamically generated RAG chain
    print(f"Received query: {query}")
    response = rag_chain.invoke({"input": query})
    answer = response["answer"]
    
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={"query": query, "answer": answer, "system_status": None}
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)