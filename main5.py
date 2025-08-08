# main.py
# Final version adapted for HackRx 6.0 submission requirements.
# This script runs a FastAPI server with the required /hackrx/run endpoint.
# For each request, it dynamically downloads a PDF from a URL, processes it in memory,
# and answers a list of questions based on its content.

import os
import fitz  # PyMuPDF
import requests
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, HttpUrl
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration & Security ---
HACKATHON_API_KEY = os.getenv("HACKATHON_API_KEY", "29ac380920ae0807a02894db4f79b819b20a6410ba3fcebd7adf0a924a01eae3")
security = HTTPBearer()

# --- Pydantic Models for API ---
class HackathonRequest(BaseModel):
    documents: HttpUrl
    questions: list[str]

class HackathonResponse(BaseModel):
    answers: list[str]

# --- FastAPI App Initialization ---
app = FastAPI(
    title="HackRx 6.0 Intelligent Query-Retrieval System",
    description="Processes a document from a URL and answers questions based on its content.",
    version="2.0.6" # Final memory optimization
)

# --- Global Components (Loaded once on startup) ---
# PERFORMANCE OPTIMIZATION: Using a smaller, more memory-efficient model loaded at startup.
print("Loading embedding model...")
try:
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L3-v2")
    print("Embedding model loaded successfully.")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    embeddings = None

print("Initializing Groq LLM...")
llm = ChatGroq(
    model_name="llama3-8b-8192", # Using a smaller, faster LLM
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)
print("Groq LLM initialized.")

# --- Prompt Engineering ---
qa_prompt_template = """
You are an expert AI assistant. Answer the user's question based *only* on the provided 'Relevant Clauses'.
If the answer is not in the context, state that clearly. Be concise.

**Relevant Clauses:**
---
{context}
---

**Question:**
{question}

**Answer:**
"""
QA_PROMPT = PromptTemplate.from_template(qa_prompt_template)

# --- Helper Functions for Dynamic Processing ---
def process_document_from_url(url: str):
    """Downloads a PDF from a URL, extracts text, chunks it, and creates a vector store."""
    if not embeddings:
        raise HTTPException(status_code=500, detail="Embedding model is not available.")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        pdf_bytes = response.content

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        # OPTIMIZATION: Process text page by page to reduce peak memory usage
        all_chunks = []
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            if page_text:
                chunks = text_splitter.create_documents([page_text], metadatas=[{"source": f"page_{page_num+1}"}])
                all_chunks.extend(chunks)

        if not all_chunks:
            raise HTTPException(status_code=400, detail="Could not extract any text from the document.")

        vector_store = Chroma.from_documents(documents=all_chunks, embedding=embeddings)
        return vector_store.as_retriever(search_kwargs={"k": 3})

    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download document: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")

# --- Security Dependency ---
def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verifies the Bearer token against the required hackathon key."""
    if credentials.scheme != "Bearer" or credentials.credentials != HACKATHON_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")
    return credentials.credentials

# --- API Endpoint Definitions ---
@app.post("/hackrx/run", response_model=HackathonResponse, tags=["Hackathon"])
async def run_submission(request: HackathonRequest, token: str = Security(verify_token)):
    """
    This endpoint receives a document URL and a list of questions.
    It processes the document on-the-fly and returns a list of answers.
    """
    print(f"Processing request for document: {request.documents}")
    
    retriever = process_document_from_url(str(request.documents))

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | QA_PROMPT
        | llm
        | StrOutputParser()
    )

    answers = []
    print(f"Answering {len(request.questions)} questions...")
    for question in request.questions:
        try:
            answer = rag_chain.invoke(question)
            answers.append(answer)
        except Exception as e:
            answers.append(f"Error processing question: {e}")
    
    print("Finished answering all questions.")
    return {"answers": answers}
