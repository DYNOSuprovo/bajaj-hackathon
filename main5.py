# main.py - Ultra minimal version for Render deployment
import os
import gc
import tempfile
import fitz  # PyMuPDF
import requests
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
HACKATHON_API_KEY = os.getenv("HACKATHON_API_KEY", "29ac380920ae0807a02894db4f79b819b20a6410ba3fcebd7adf0a924a01eae3")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
security = HTTPBearer()

# Pydantic Models
class HackathonRequest(BaseModel):
    documents: HttpUrl
    questions: list[str]

class HackathonResponse(BaseModel):
    answers: list[str]

# FastAPI App
app = FastAPI(
    title="HackRx 6.0 Minimal Q&A System",
    description="Lightweight document Q&A system",
    version="1.0.0"
)

# Simple text extraction
def extract_text_from_url(url: str) -> str:
    """Extract text from PDF URL with minimal memory usage"""
    try:
        # Download PDF
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Use temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(response.content)
            tmp_file.flush()
            
            # Extract text
            doc = fitz.open(tmp_file.name)
            text = ""
            
            # Limit to first 10 pages for memory
            max_pages = min(len(doc), 10)
            for page_num in range(max_pages):
                page = doc[page_num]
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.get_text()
            
            doc.close()
            os.unlink(tmp_file.name)
            
            return text[:50000]  # Limit text length
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process document: {str(e)}")

# Simple Q&A using Groq
def answer_question(text: str, question: str) -> str:
    """Simple Q&A using Groq API directly"""
    try:
        import json
        
        # Find relevant text (simple keyword search)
        question_lower = question.lower()
        text_lines = text.split('\n')
        relevant_lines = []
        
        for line in text_lines:
            if any(word in line.lower() for word in question_lower.split() if len(word) > 3):
                relevant_lines.append(line.strip())
        
        # Use first 10 relevant lines
        context = '\n'.join(relevant_lines[:10]) if relevant_lines else text[:2000]
        
        # Direct API call to Groq
        headers = {
            'Authorization': f'Bearer {GROQ_API_KEY}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': 'llama3-8b-8192',
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are a helpful assistant. Answer questions based only on the provided context. Be concise.'
                },
                {
                    'role': 'user',
                    'content': f'Context: {context}\n\nQuestion: {question}\n\nAnswer:'
                }
            ],
            'max_tokens': 300,
            'temperature': 0
        }
        
        response = requests.post(
            'https://api.groq.com/openai/v1/chat/completions',
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            return f"Error: Unable to get answer from API"
            
    except Exception as e:
        return f"Error processing question: {str(e)}"

# Security
def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != HACKATHON_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return credentials.credentials

# Main endpoint
@app.post("/hackrx/run", response_model=HackathonResponse)
async def run_submission(request: HackathonRequest, token: str = Security(verify_token)):
    """Process document and answer questions"""
    
    if not GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="Groq API key not configured")
    
    try:
        # Extract text
        text = extract_text_from_url(str(request.documents))
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text extracted from document")
        
        # Answer questions
        answers = []
        max_questions = min(len(request.questions), 5)  # Limit questions
        
        for question in request.questions[:max_questions]:
            answer = answer_question(text, question)
            answers.append(answer)
            gc.collect()  # Force cleanup
        
        return {"answers": answers}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"message": "HackRx 6.0 Q&A System"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
