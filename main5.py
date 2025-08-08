# main.py
# Memory-optimized version for HackRx 6.0 submission v2
# Implements more aggressive memory management and stricter limits for free-tier deployment.

import os
import gc
import tempfile
import fitz  # PyMuPDF
import requests
import logging
from fastapi import FastAPI, HTTPException, Security, Request
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
# Configure logging to provide clear output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from a .env file if it exists
load_dotenv()

# --- Security and API Key Configuration ---
# It's good practice to fetch the API key from environment variables.
# The provided key is used as a default if the environment variable is not set.
HACKATHON_API_KEY = os.getenv("HACKATHON_API_KEY", "29ac380920ae0807a02894db4f79b819b20a6410ba3fcebd7adf0a924a01eae3")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize the security scheme for bearer token authentication
security = HTTPBearer()

# --- Pydantic Models for Request/Response Validation ---
class HackathonRequest(BaseModel):
    """Defines the expected structure of the incoming request body."""
    documents: HttpUrl = Field(..., description="URL of the PDF document to process.")
    questions: list[str] = Field(..., description="A list of questions to answer based on the document.")

class HackathonResponse(BaseModel):
    """Defines the structure of the response."""
    answers: list[str]

# --- FastAPI Application Setup ---
app = FastAPI(
    title="HackRx 6.0 Memory-Optimized System",
    description="A memory-efficient document processing and Q&A system designed to run on resource-constrained environments.",
    version="3.0.1"
)

# --- Lazy Loading for AI Models ---
# Global variables to hold the models. They are initialized to None and loaded only when first needed.
# This prevents loading large models into memory on application startup, which is critical for free tiers.
_embeddings = None
_llm = None

def get_embeddings():
    """
    Lazily loads the HuggingFace embedding model.
    This function is called only when embeddings are required, reducing initial memory footprint.
    """
    global _embeddings
    if _embeddings is None:
        logger.info("Loading embedding model: all-MiniLM-L6-v2...")
        try:
            # Using a small, efficient model is key for memory savings.
            _embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},  # Explicitly use CPU
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Fatal: Could not load embedding model. Error: {e}")
            raise HTTPException(status_code=500, detail="Failed to load a critical component (embedding model).")
    return _embeddings

def get_llm():
    """
    Lazily loads the ChatGroq LLM.
    This ensures the LLM is only loaded when a question is being processed.
    """
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
            max_tokens=350  # Limit response length to conserve resources
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

# --- Core Document Processing Logic ---
def process_document_stream(url: str):
    """
    Downloads, processes, and embeds a PDF document with aggressive memory management.
    - Uses a temporary file to avoid holding the entire PDF in memory.
    - Processes a limited number of pages to prevent timeouts and memory overload.
    - Creates a Chroma vector store in-memory.
    """
    logger.info(f"Starting document processing for URL: {url}")
    try:
        # Stream the download to handle large files without high memory usage.
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(response.content)
            tmp_filepath = tmp_file.name
        
        logger.info("Document downloaded successfully.")
        
        doc = fitz.open(tmp_filepath)
        
        # --- Resource Limiting ---
        max_pages = 15  # Reduced page limit
        total_pages = min(len(doc), max_pages)
        logger.info(f"Processing {total_pages} of {len(doc)} pages.")

        full_text = "".join([page.get_text() for page in doc.pages(0, total_pages - 1)])
        doc.close()

        if not full_text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from the document.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=450,  # Smaller chunks for faster processing
            chunk_overlap=40
        )
        text_chunks = text_splitter.split_text(full_text)
        
        # --- Resource Limiting ---
        max_chunks = 75 # Stricter chunk limit
        if len(text_chunks) > max_chunks:
            logger.warning(f"Document has {len(text_chunks)} chunks. Limiting to {max_chunks} for memory management.")
            text_chunks = text_chunks[:max_chunks]

        documents = [Document(page_content=chunk) for chunk in text_chunks]
        
        logger.info(f"Creating vector store from {len(documents)} document chunks.")
        embeddings = get_embeddings()
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=None  # In-memory only
        )
        
        # Clean up the temporary file immediately
        os.unlink(tmp_filepath)
        
        logger.info("Vector store created. Returning retriever.")
        # --- Resource Limiting ---
        # Retrieve only the top 1 document to reduce LLM context size and processing time.
        return vector_store.as_retriever(search_kwargs={"k": 1})

    except requests.RequestException as e:
        logger.error(f"Download failed for {url}. Error: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download the document. Please check the URL. Error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during document processing: {e}", exc_info=True)
        # Clean up temp file in case of error
        if 'tmp_filepath' in locals() and os.path.exists(tmp_filepath):
            os.unlink(tmp_filepath)
        raise HTTPException(status_code=500, detail=f"An internal error occurred while processing the document.")

# --- Security Dependency ---
def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verifies the provided bearer token against the configured API key."""
    if credentials.scheme != "Bearer" or credentials.credentials != HACKATHON_API_KEY:
        logger.warning("Failed authentication attempt.")
        raise HTTPException(status_code=403, detail="Invalid or missing API key.")
    return credentials.credentials

# --- API Endpoint ---
@app.post("/hackrx/run", response_model=HackathonResponse, tags=["Hackathon"])
async def run_submission(request: HackathonRequest, token: str = Security(verify_token)):
    """
    The main endpoint for the hackathon. It processes a document and answers questions.
    """
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
        # --- Resource Limiting ---
        max_questions = 5 # Limit number of questions processed per request
        questions_to_process = request.questions[:max_questions]
        
        logger.info(f"Processing {len(questions_to_process)} questions.")
        for i, question in enumerate(questions_to_process):
            try:
                logger.info(f"Invoking RAG chain for question {i+1}...")
                answer = rag_chain.invoke(question[:500]) # Limit question length
                answers.append(answer)
                logger.info(f"Successfully processed question {i+1}.")
            except Exception as e:
                logger.error(f"Error processing question '{question}': {e}", exc_info=True)
                answers.append("Error: Could not process this question.")
        
        logger.info("Request completed successfully.")
        return HackathonResponse(answers=answers)

    except HTTPException:
        # Re-raise HTTPExceptions to let FastAPI handle them
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
    """A simple health check endpoint to confirm the service is up."""
    return {"status": "healthy"}

# --- Application Startup Event ---
@app.on_event("startup")
async def startup_event():
    """Logs a message when the application starts."""
    logger.info("Application startup complete. Ready to accept requests.")

# --- Main entry point for running with uvicorn ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
