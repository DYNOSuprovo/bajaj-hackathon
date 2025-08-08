# main.py
# Memory-optimized version for HackRx 6.0 submission v3
# Implements ultra-aggressive memory management, stricter limits, and request timeouts for free-tier deployment.

import os
import asyncio
import tempfile
import fitz  # PyMuPDF
import requests
import logging
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

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# --- Security and API Key Configuration ---
HACKATHON_API_KEY = os.getenv("HACKATHON_API_KEY", "29ac380920ae0807a02894db4f79b819b20a6410ba3fcebd7adf0a924a01eae3")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
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
    version="3.0.2"
)

# --- Lazy Loading for AI Models ---
_embeddings = None
_llm = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        logger.info("Loading embedding model: all-MiniLM-L6-v2...")
        try:
            _embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Fatal: Could not load embedding model. Error: {e}")
            raise HTTPException(status_code=500, detail="Failed to load a critical component (embedding model).")
    return _embeddings

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

# --- Core Document Processing Logic (Page-by-Page) ---
def process_document_stream(url: str):
    logger.info(f"Starting document processing for URL: {url}")
    tmp_filepath = None
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(response.content)
            tmp_filepath = tmp_file.name
        
        logger.info("Document downloaded successfully.")
        doc = fitz.open(tmp_filepath)
        
        # --- Ultra-Aggressive Resource Limiting ---
        max_pages = 10
        total_pages = min(len(doc), max_pages)
        logger.info(f"Processing {total_pages} of {len(doc)} pages.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=40
        )
        
        all_text_chunks = []
        # Process page by page to reduce peak memory
        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            page_text = page.get_text("text")
            if page_text.strip():
                chunks = text_splitter.split_text(page_text)
                all_text_chunks.extend(chunks)
        
        doc.close()

        if not all_text_chunks:
            raise HTTPException(status_code=400, detail="No text could be extracted from the document.")

        # --- Stricter Chunk Limiting ---
        max_chunks = 50
        if len(all_text_chunks) > max_chunks:
            logger.warning(f"Document has {len(all_text_chunks)} chunks. Limiting to {max_chunks} for memory management.")
            all_text_chunks = all_text_chunks[:max_chunks]

        documents = [Document(page_content=chunk) for chunk in all_text_chunks]
        
        logger.info(f"Creating vector store from {len(documents)} document chunks.")
        embeddings = get_embeddings()
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=None
        )
        
        logger.info("Vector store created. Returning retriever.")
        return vector_store.as_retriever(search_kwargs={"k": 1})

    except requests.RequestException as e:
        logger.error(f"Download failed for {url}. Error: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download the document. Error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during document processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred while processing the document.")
    finally:
        # Ensure temporary file is always cleaned up
        if tmp_filepath and os.path.exists(tmp_filepath):
            os.unlink(tmp_filepath)

# --- Security Dependency ---
def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != HACKATHON_API_KEY:
        logger.warning("Failed authentication attempt.")
        raise HTTPException(status_code=403, detail="Invalid or missing API key.")
    return credentials.credentials

# --- API Endpoint with Timeouts ---
@app.post("/hackrx/run", response_model=HackathonResponse, tags=["Hackathon"])
async def run_submission(request: HackathonRequest, token: str = Security(verify_token)):
    logger.info("Received new request for /hackrx/run.")
    try:
        retriever = process_document_stream(str(request.documents))
        llm = get_llm()

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | QA_PROMPT
            | llm
            | StrOutputParser()
        )

        answers = []
        max_questions = 5
        questions_to_process = request.questions[:max_questions]
        
        logger.info(f"Processing {len(questions_to_process)} questions.")
        for i, question in enumerate(questions_to_process):
            try:
                logger.info(f"Invoking RAG chain for question {i+1}...")
                # Add a timeout to each question to prevent the entire request from hanging
                answer = await asyncio.wait_for(
                    rag_chain.ainvoke(question[:500]),
                    timeout=25.0
                )
                answers.append(answer)
                logger.info(f"Successfully processed question {i+1}.")
            except asyncio.TimeoutError:
                logger.error(f"Timeout processing question '{question}'")
                answers.append("Error: The request timed out while processing this question.")
            except Exception as e:
                logger.error(f"Error processing question '{question}': {e}", exc_info=True)
                answers.append("Error: Could not process this question due to an internal error.")
        
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
