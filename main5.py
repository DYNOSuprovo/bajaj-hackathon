# main.py
# Memory-optimized version for HackRx 6.0 submission v4
# FINAL VERSION: Offloads embedding generation to the Hugging Face Inference API to avoid local memory crashes.

import os
import asyncio
import tempfile
import fitz  # PyMuPDF
import requests
import logging
import numpy as np
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, HttpUrl
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# --- Security and API Key Configuration ---
HACKATHON_API_KEY = os.getenv("HACKATHON_API_KEY", "29ac380920ae0807a02894db4f79b819b20a6410ba3fcebd7adf0a924a01eae3")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_API_TOKEN = os.getenv("HF_API_TOKEN") # <-- ACTION REQUIRED: Add this to your Render environment variables
EMBEDDING_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

security = HTTPBearer()

# --- Pydantic Models ---
class HackathonRequest(BaseModel):
    documents: HttpUrl = Field(..., description="URL of the PDF document to process.")
    questions: list[str] = Field(..., description="A list of questions to answer based on the document.")

class HackathonResponse(BaseModel):
    answers: list[str]

# --- FastAPI Application Setup ---
app = FastAPI(
    title="HackRx 6.0 Memory-Optimized System",
    description="A memory-efficient document processing and Q&A system designed to run on resource-constrained environments.",
    version="4.0.0"
)

# --- Lazy Loading for LLM ---
_llm = None

def get_llm():
    global _llm
    if _llm is None:
        if not GROQ_API_KEY:
            logger.error("Fatal: GROQ_API_KEY is not set in environment variables.")
            raise HTTPException(status_code=500, detail="Server configuration error: Missing Groq API key.")
        logger.info("Initializing Groq LLM: llama3-8b-8192...")
        _llm = ChatGroq(
            model_name="llama3-8b-8192",
            temperature=0,
            api_key=GROQ_API_KEY,
            max_tokens=350
        )
        logger.info("Groq LLM initialized.")
    return _llm

# --- Optimized RAG Prompt Template ---
qa_prompt_template = """SYSTEM: You are a helpful assistant. Answer the user's question based *only* on the context provided. Be concise and direct. If the context does not contain the answer, say "The answer is not available in the provided document."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
QA_PROMPT = PromptTemplate.from_template(qa_prompt_template)


# --- Embedding Generation via API ---
def get_embeddings_from_api(texts: list[str]) -> np.ndarray:
    """Gets embeddings for a list of texts using the Hugging Face Inference API."""
    if not HF_API_TOKEN:
        raise HTTPException(status_code=500, detail="Server configuration error: Missing Hugging Face API Token.")
    
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    
    try:
        response = requests.post(
            EMBEDDING_API_URL,
            headers=headers,
            json={"inputs": texts, "options": {"wait_for_model": True}}
        )
        response.raise_for_status()
        embeddings = np.array(response.json())
        # Ensure the embeddings are 2D
        if embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, axis=0)
        return embeddings
    except requests.RequestException as e:
        logger.error(f"API call to get embeddings failed: {e}")
        raise HTTPException(status_code=503, detail="The embedding service is currently unavailable.")


# --- Core Document Processing Logic ---
def process_document_and_get_chunks(url: str) -> list[str]:
    """Downloads and extracts text chunks from a PDF."""
    logger.info(f"Starting document processing for URL: {url}")
    tmp_filepath = None
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(response.content)
            tmp_filepath = tmp_file.name
        
        doc = fitz.open(tmp_filepath)
        
        max_pages = 15
        total_pages = min(len(doc), max_pages)
        logger.info(f"Processing {total_pages} of {len(doc)} pages.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
        
        all_text_chunks = []
        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            page_text = page.get_text("text")
            if page_text.strip():
                chunks = text_splitter.split_text(page_text)
                all_text_chunks.extend(chunks)
        
        doc.close()

        if not all_text_chunks:
            raise HTTPException(status_code=400, detail="No text could be extracted from the document.")

        max_chunks = 75
        if len(all_text_chunks) > max_chunks:
            logger.warning(f"Document has {len(all_text_chunks)} chunks. Limiting to {max_chunks}.")
            all_text_chunks = all_text_chunks[:max_chunks]
        
        return all_text_chunks

    finally:
        if tmp_filepath and os.path.exists(tmp_filepath):
            os.unlink(tmp_filepath)


# --- Security Dependency ---
def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != HACKATHON_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key.")
    return credentials.credentials


# --- API Endpoint ---
@app.post("/hackrx/run", response_model=HackathonResponse, tags=["Hackathon"])
async def run_submission(request: HackathonRequest, token: str = Security(verify_token)):
    logger.info("Received new request for /hackrx/run.")
    try:
        # 1. Process Document
        text_chunks = process_document_and_get_chunks(str(request.documents))
        
        # 2. Get Embeddings for all chunks via API
        logger.info(f"Getting embeddings for {len(text_chunks)} chunks via API...")
        chunk_embeddings = get_embeddings_from_api(text_chunks)
        logger.info("Chunk embeddings received.")

        llm = get_llm()
        answers = []
        
        questions_to_process = request.questions[:5] # Limit questions
        
        for i, question in enumerate(questions_to_process):
            logger.info(f"Processing question {i+1}: {question[:80]}...")
            
            # 3. Get embedding for the current question
            question_embedding = get_embeddings_from_api([question])
            
            # 4. Find the most relevant chunk (manual retrieval)
            similarities = cosine_similarity(question_embedding, chunk_embeddings)
            most_relevant_chunk_index = np.argmax(similarities)
            context = text_chunks[most_relevant_chunk_index]
            
            # 5. Build and invoke the RAG chain for this question
            prompt = QA_PROMPT.format(context=context, question=question)
            
            try:
                answer = await asyncio.wait_for(
                    llm.ainvoke(prompt),
                    timeout=25.0
                )
                answers.append(answer.content)
                logger.info(f"Successfully answered question {i+1}.")
            except asyncio.TimeoutError:
                logger.error(f"Timeout processing question '{question}'")
                answers.append("Error: The request timed out while processing this question.")
        
        logger.info("Request completed successfully.")
        return HackathonResponse(answers=answers)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"A critical error occurred during the request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected server error occurred.")


# --- Health Check and Root Endpoints ---
@app.get("/", include_in_schema=False)
def root():
    return {"message": "HackRx 6.0 Memory-Optimized System is running. See /docs for API documentation."}

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy"}


# --- Application Startup Event ---
@app.on_event("startup")
async def startup_event():
    logger.info("Application startup complete. Ready to accept requests.")


# --- Main entry point ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
