# main.py
import os
import fitz 
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

load_dotenv()

HACKATHON_API_KEY = os.getenv("HACKATHON_API_KEY", "29ac380920ae0807a02894db4f79b819b20a6410ba3fcebd7adf0a924a01eae3")
security = HTTPBearer()

class HackathonRequest(BaseModel):
    documents: HttpUrl
    questions: list[str]

class HackathonResponse(BaseModel):
    answers: list[str]

app = FastAPI(
    title="HackRx 6.0 Intelligent Query-Retrieval System",
    description="Processes a document from a URL and answers questions based on its content.",
    version="2.0.1" # Updated version
)

print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Embedding model loaded.")

print("Initializing Groq LLM...")
llm = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)
print("Groq LLM initialized.")

qa_prompt_template = """
You are an expert AI assistant for answering questions based on a provided document.
Your task is to answer the user's question accurately and concisely, using *only* the information from the 'Relevant Clauses'.
Do not use any external knowledge. If the answer is not found in the provided context, state that clearly.

**Relevant Clauses:**
---
{context}
---

**Question:**
{question}

**Answer:**
"""
QA_PROMPT = PromptTemplate.from_template(qa_prompt_template)
def process_document_from_url(url: str):
    """Downloads a PDF from a URL, extracts text, chunks it, and creates a vector store."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        pdf_bytes = response.content
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        documents = [Document(page_content=chunk) for chunk in chunks]
        vector_store = Chroma.from_documents(documents=documents, embedding=embeddings)
        return vector_store.as_retriever(search_kwargs={"k": 5})

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

# --- Main API Endpoint ---
@app.post("/hackrx/run", response_model=HackathonResponse)
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

