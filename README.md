# RAG Document Assistant

This project is a Retrieval-Augmented Generation (RAG) application that allows you to interact with your PDF documents using Google's Gemini LLM. It offers two different user interfaces:
1. A **Streamlit** based frontend for a highly interactive and easy-to-use visual chat interface.
2. A **FastAPI** based web server with a Jinja2 HTML template frontend.

## Features
- **Upload Multiple PDFs**: Upload one or more PDF documents to act as the knowledge base.
- **RAG-powered Q&A**: Employs LangChain and ChromaDB to chunk, embed, and retrieve relevant document sections to answer your queries accurately.
- **Google Gemini LLM**: Uses `gemini-2.5-flash` for high-quality, fast generation.
- **HuggingFace Embeddings**: Uses `sentence-transformers/all-MiniLM-L6-v2` for efficient local document embedding.
- **Dual Interfaces**: Choose between a fast interactive Streamlit app or a traditional FastAPI web server.

## Tech Stack
- **Python Frameworks**: [Streamlit](https://streamlit.io/), [FastAPI](https://fastapi.tiangolo.com/)
- **LLM & Tooling**: [LangChain](https://python.langchain.com/), Google Gemini (`langchain-google-genai`)
- **Vector Store**: [ChromaDB](https://www.trychroma.com/)
- **Embeddings**: HuggingFace (`sentence-transformers`)
- **Document Processing**: `pypdf`, LangChain Text Splitters

## Prerequisites
- Python 3.8+
- [Google Gemini API Key](https://aistudio.google.com/)

## Setup & Installation

1. **Clone the repository** (or download the source code):
   ```bash
   git clone <repository-url>
   cd RAG
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate

   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**:
   Create a `.env` file in the root directory and add your Google Gemini API Key:
   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## Running the Application

### Option 1: Streamlit Interface (Recommended)
The Streamlit interface provides a rich, interactive chat UI.
```bash
streamlit run app.py
```
This will automatically open the app in your default web browser (usually at `http://localhost:8501`).

### Option 2: FastAPI Interface
The FastAPI version uses an HTML template to serve the application.
```bash
uvicorn main:app --reload
```
Once started, open your web browser and navigate to `http://localhost:8000`.

## Project Structure
- `app.py`: Streamlit application entry point.
- `main.py`: FastAPI server entry point.
- `templates/`: Contains HTML files (e.g., `index.html`) used by FastAPI.
- `uploads/`: Temporary storage for uploaded PDF files (FastAPI).
- `requirements.txt`: Python dependencies.
- `render.yaml`: Configuration file for deployment on Render.
- `rag.ipynb`: Jupyter notebook for experimenting with the RAG pipeline.

## Notes
- To support ChromaDB on platforms with older SQLite versions (like Render), the project uses `pysqlite3-binary` to hot-swap the system SQLite module.
