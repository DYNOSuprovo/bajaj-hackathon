# main.py
# Memory-optimized version for HackRx 6.0 submission
# Implements aggressive memory management and streaming processing

import os
import gc
import tempfile
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
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- Configuration & Security ---
HACKATHON_API_KEY = os.getenv("HACKATHON_API_KEY", "29ac380920ae0807a02894db4f79b819b20a6410ba3fcebd7adf0a924a01eae3")
security = HTTPBearer()

# Memory monitoring (simplified without psutil)
def log_memory_usage(operation: str):
    """Log operation (memory monitoring disabled)"""
    logger.info(f"{operation}: Processing...")

# --- Pydantic Models ---
class HackathonRequest(BaseModel):
    documents: HttpUrl
    questions: list[str]

class HackathonResponse(BaseModel):
    answers: list[str]

# --- FastAPI App ---
app = FastAPI(
    title="HackRx 6.0 Memory-Optimized System",
    description="Memory-efficient document processing and Q&A system.",
    version="3.0.0"
)

# --- Lazy Loading Components ---
_embeddings = None
_llm = None

def get_embeddings():
    """Lazy load embeddings model"""
    global _embeddings
    if _embeddings is None:
        logger.info("Loading embedding model...")
        try:
            # Use an even smaller model for better memory efficiency
            _embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",  # Smaller than paraphrase-MiniLM-L3-v2
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            log_memory_usage("Embeddings loaded")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise HTTPException(status_code=500, detail="Failed to load embedding model")
    return _embeddings

def get_llm():
    """Lazy load LLM"""
    global _llm
    if _llm is None:
        logger.info("Initializing Groq LLM...")
        _llm = ChatGroq(
            model_name="llama3-8b-8192",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY"),
            max_tokens=512  # Limit response length
        )
        log_memory_usage("LLM initialized")
    return _llm

# --- Optimized Prompt ---
qa_prompt_template = """Answer based only on the context. Be concise.

Context: {context}

Question: {question}

Answer:"""

QA_PROMPT = PromptTemplate.from_template(qa_prompt_template)

# --- Memory-Optimized Document Processing ---
def process_document_stream(url: str, max_pages: int = 20):
    """Process document with aggressive memory management"""
    log_memory_usage("Starting document processing")
    
    try:
        # Stream download with size limit
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Use temporary file to avoid keeping full PDF in memory
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            size = 0
            max_size = 50 * 1024 * 1024  # 50MB limit
            
            for chunk in response.iter_content(chunk_size=8192):
                size += len(chunk)
                if size > max_size:
                    raise HTTPException(status_code=400, detail="Document too large (>50MB)")
                tmp_file.write(chunk)
            
            tmp_file.flush()
            log_memory_usage("Document downloaded")
            
            # Process PDF with memory management
            doc = fitz.open(tmp_file.name)
            
            # Limit pages processed
            total_pages = min(len(doc), max_pages)
            logger.info(f"Processing {total_pages} pages (limited from {len(doc)})")
            
            # Process in smaller batches
            all_text_chunks = []
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,  # Smaller chunks
                chunk_overlap=50,
                length_function=len
            )
            
            batch_size = 5  # Process 5 pages at a time
            for start_page in range(0, total_pages, batch_size):
                end_page = min(start_page + batch_size, total_pages)
                batch_text = ""
                
                # Extract text from batch of pages
                for page_num in range(start_page, end_page):
                    page = doc[page_num]
                    page_text = page.get_text()
                    if page_text.strip():
                        batch_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                    
                    # Clear page from memory
                    del page_text
                
                # Split batch text into chunks
                if batch_text.strip():
                    batch_chunks = text_splitter.split_text(batch_text)
                    all_text_chunks.extend(batch_chunks)
                
                # Clear batch text
                del batch_text
                gc.collect()
            
            doc.close()
            
            # Clean up temp file
            os.unlink(tmp_file.name)
            
            log_memory_usage(f"Text extracted: {len(all_text_chunks)} chunks")
            
            if not all_text_chunks:
                raise HTTPException(status_code=400, detail="No text extracted from document")
            
            # Limit number of chunks to manage memory
            max_chunks = 100
            if len(all_text_chunks) > max_chunks:
                all_text_chunks = all_text_chunks[:max_chunks]
                logger.info(f"Limited to {max_chunks} chunks for memory management")
            
            # Create documents
            documents = [Document(page_content=chunk) for chunk in all_text_chunks]
            del all_text_chunks
            gc.collect()
            
            # Create vector store with memory management
            embeddings = get_embeddings()
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=None  # Keep in memory, don't persist
            )
            
            del documents
            gc.collect()
            log_memory_usage("Vector store created")
            
            return vector_store.as_retriever(search_kwargs={"k": 2})  # Reduced k for memory
            
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Download failed: {str(e)[:100]}")
    except Exception as e:
        logger.error(f"Document processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)[:100]}")

# --- Security ---
def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != HACKATHON_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return credentials.credentials

# --- Optimized Endpoint ---
@app.post("/hackrx/run", response_model=HackathonResponse, tags=["Hackathon"])
async def run_submission(request: HackathonRequest, token: str = Security(verify_token)):
    """Memory-optimized document Q&A endpoint"""
    log_memory_usage("Request started")
    
    try:
        # Process document
        retriever = process_document_stream(str(request.documents))
        
        # Get LLM
        llm = get_llm()
        
        # Create RAG chain
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | QA_PROMPT
            | llm
            | StrOutputParser()
        )
        
        # Process questions with memory management
        answers = []
        max_questions = min(len(request.questions), 10)  # Limit questions
        
        for i, question in enumerate(request.questions[:max_questions]):
            try:
                logger.info(f"Processing question {i+1}/{max_questions}")
                answer = rag_chain.invoke(question[:500])  # Limit question length
                answers.append(answer)
                
                # Force garbage collection between questions
                if i % 2 == 1:
                    gc.collect()
                    log_memory_usage(f"After question {i+1}")
                    
            except Exception as e:
                logger.error(f"Question {i+1} error: {e}")
                answers.append(f"Error: {str(e)[:100]}")
        
        # Clean up
        del retriever, rag_chain
        gc.collect()
        log_memory_usage("Request completed")
        
        return {"answers": answers}
        
    except Exception as e:
        logger.error(f"Request failed: {e}")
        # Force cleanup on error
        gc.collect()
        raise

# --- Health Check ---
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Service is running"
    }

# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    """Log startup memory usage"""
    log_memory_usage("Application startup")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
