import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Form
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

rag_chain = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager that initializes the heavy RAG logic 
    (PDF loading, embedding, and vector store) exactly once when the server starts.
    """
    global rag_chain
    print("Starting up... Loading PDF and initializing Vector Store...")
    
    # 1. Load the document
    loader = PyPDFLoader(r"D:\AI Engineering\RAG\iso27001.pdf")
    documents = loader.load()

    # 2. Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # 3. Create Embeddings and Vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 4. Initialize LLM (Gemini)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    # 5. Define System Prompt & Create RAG Chain
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
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    print("Startup complete. RAG chain is ready to take questions!")
    
    yield
    print("Shutting down model...")

# Initialize FastAPI with the lifespan events
app = FastAPI(lifespan=lifespan)
# Configure Jinja2 templates directory
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    """Serve the basic HTML frontend page on a GET request."""
    return templates.TemplateResponse(request=request, name="index.html", context={"query": None, "answer": None})

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
                "answer": "System is still initializing. Please try again in a few seconds."
            }
         )

    # Invoke the pre-initialized RAG chain
    print(f"Received query: {query}")
    response = rag_chain.invoke({"input": query})
    answer = response["answer"]
    
    return templates.TemplateResponse(request=request, name="index.html", context={"query": query, "answer": answer})

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)