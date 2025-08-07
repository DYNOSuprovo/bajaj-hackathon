from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get API key from environment
API_KEY = os.getenv("APP_API_KEY")
if not API_KEY:
    logger.warning("APP_API_KEY is missing from .env. Defaulting to 'mydefaultkey'.")
    API_KEY = "mydefaultkey"

# FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware for API key check
@app.middleware("http")
async def api_key_check(request: Request, call_next):
    client_key = request.headers.get("x-api-key")
    logger.info(f"From .env: {API_KEY} | From request: {client_key}")
    if client_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key.")
    return await call_next(request)

# Define request body
class UserQuery(BaseModel):
    query: str

# Load vector DB and LLM
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="./clause_vector_store", embedding_function=embedding)
retriever = vectordb.as_retriever(search_kwargs={"k": 5})
llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

# Prompt template
template = """
You are an expert IRDAI health insurance claim advisor. Given the user's query and the policy clauses, determine:

1. Should the claim be accepted or rejected?
2. What is the amount to be reimbursed?
3. Which clauses support this decision?

Respond in JSON like this:
{{
  "decision": "Accepted or Rejected",
  "amount": <number>,
  "reason": "<short reason>",
  "clause_references": ["clause1", "clause2"],
  "matched_clauses": ["<actual text of clause1>", "<clause2>"]
}}

QUERY: {query}
CLAUSES:
{context}
"""

prompt = PromptTemplate(input_variables=["context", "query"], template=template)
rag_chain = (
    {"context": retriever | RunnablePassthrough(), "query": RunnablePassthrough()}
    | prompt
    | llm
)

# Main route
@app.post("/predict")
async def generate_response(user_query: UserQuery):
    query = user_query.query
    context_docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in context_docs[:5])

    if not context_docs:
        return {
            "decision": "Rejected",
            "amount": 0.0,
            "reason": "No matching clauses found in the policy document.",
            "clause_references": [],
            "matched_clauses": []
        }

    result = rag_chain.invoke(query)
    return result

# Health check route
@app.get("/health")
def health():
    return {"status": "ok"}
