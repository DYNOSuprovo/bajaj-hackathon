# main.py
# This script runs a FastAPI web server to handle user queries.
# It loads the pre-built vector database, connects to the Groq API for LLM reasoning,
# and uses LangChain to orchestrate the Retrieval-Augmented Generation (RAG) pipeline.

from fastapi import FastAPI
from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
# --- CORRECTED IMPORT ---
# JsonOutputParser and StrOutputParser were moved to 'langchain_core' in a recent update.
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
# --- END CORRECTION ---
from dotenv import load_dotenv
import os

# Load environment variables from the .env file.
# This is where your GROQ_API_KEY should be stored.
load_dotenv()

# --- Pydantic Models for API Input and Output ---
# Defines the structure of the incoming request JSON.
class QueryRequest(BaseModel):
    query: str

# Defines the structure and validation for the final JSON output.
# Adding descriptions helps the LLM generate better-formatted responses.
class Justification(BaseModel):
    decision: str = Field(description="The final decision, either 'Approved' or 'Rejected'.")
    amount: float = Field(description="The payable amount. Should be 0.0 if rejected.")
    reason: str = Field(description="A clear, concise justification for the decision based on policy clauses.")
    clause_references: list[str] = Field(description="A list of specific clauses and their source documents that support the decision.")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="HackRx 6.0 LLM Document Processor (API Version)",
    description="An API-powered system for processing insurance policy documents using RAG.",
    version="1.0.1" # Updated version
)

# --- Global Component Loading (on application startup) ---
# These components are loaded once to be reused for every API call, improving performance.
try:
    print("Loading embedding model and vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    
    # The retriever is the core component for searching the vector database.
    # We configure it to return the top 10 most relevant document chunks.
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    print("Components loaded successfully.")
except Exception as e:
    print(f"Error loading components: {e}")
    retriever = None

# Initialize the Groq LLM.
# We use the powerful Llama3 70B model for high-quality reasoning.
# Temperature=0 makes the output deterministic and factual, which is crucial for this task.
llm = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY") # Securely load API key
)

# --- Prompt Engineering: Defining the LLM's instructions ---

# Prompt Template 1: For parsing the user's raw query into a structured format.
parser_prompt_template = """
You are an expert at parsing and structuring user queries related to insurance claims.
Based on the following query, extract the key details into a valid JSON object.
The query might be vague, so infer the values to the best of your ability.

Query:
{query}

Extracted JSON:
"""
PARSER_PROMPT = PromptTemplate.from_template(parser_prompt_template)

# Prompt Template 2: The main prompt for reasoning and generating the final decision.
final_prompt_template = """
You are an expert insurance claims adjudicator. Your task is to make a decision based *only* on the provided policy clauses.
Do not use any external knowledge or make assumptions beyond the provided context.

**User's Query Details (Structured):**
{structured_query}

**Relevant Policy Clauses (Context):**
---
{context}
---

**Your Task:**
1.  Carefully analyze the user's query details against the provided policy clauses.
2.  Pay close attention to waiting periods, exclusions, and specific procedure coverages.
3.  Determine if the claim should be 'Approved' or 'Rejected'.
4.  If approved, state the payable amount. If rejected, the amount must be 0.0.
5.  Provide a clear, step-by-step justification for your decision, referencing the specific source documents and clauses that support your conclusion (e.g., "as per Clause X in BAJHLIP23020V012223.pdf").
6.  Extract specific clause references that support your decision.

**IMPORTANT: You must return a JSON object with exactly these field names:**
- "decision": either "Approved" or "Rejected"
- "amount": the payable amount (number, use 0.0 if rejected)
- "reason": your detailed justification
- "clause_references": a list of specific clauses and source documents

**Example format:**
{{
  "decision": "Approved",
  "amount": 25000.0,
  "reason": "Detailed justification here...",
  "clause_references": ["Clause 15 from BAJHLIP23020V012223.pdf", "Code-Excl03 waiting period clause"]
}}

**Final JSON Output:**
"""
FINAL_PROMPT = PromptTemplate.from_template(final_prompt_template)


# --- LangChain Expression Language (LCEL) Chains ---
# LCEL allows us to declaratively chain components together into a processing pipeline.

# Simplified approach: Create a function to handle the complex logic
def process_query_pipeline(query: str):
    """Process the query through the complete RAG pipeline"""
    
    # Step 1: Structure the query
    structured_query = structuring_chain.invoke(query)
    
    # Step 2: Retrieve relevant context using original query
    context_docs = retriever.invoke(query)
    
    # Step 3: Format context for the prompt
    context = "\n\n".join([doc.page_content for doc in context_docs])
    
    # Step 4: Create the final prompt input
    prompt_input = {
        "structured_query": structured_query,
        "context": context
    }
    
    # Step 5: Get the final decision from LLM
    final_response = (FINAL_PROMPT | llm | JsonOutputParser(pydantic_object=Justification)).invoke(prompt_input)
    
    return final_response

# Chain 1: This chain takes the raw query, sends it to the LLM with the parser prompt,
# and outputs the structured string.
structuring_chain = PARSER_PROMPT | llm | StrOutputParser()

# --- API Endpoint Definition ---
@app.post("/process-query", response_model=Justification)
async def process_query(request: QueryRequest):
    """
    Processes a natural language query against the indexed insurance documents.
    This endpoint orchestrates the full RAG pipeline to provide a structured,
    justified answer based on the provided policy PDFs.
    """
    if not retriever:
        return {"error": "Vector store not initialized. Please run build_index.py first."}
    
    # Process the query through our custom pipeline function
    # This executes the entire multi-step reasoning process.
    result = process_query_pipeline(request.query)
    return result

# To run this application:
# 1. Make sure you have a .env file with your GROQ_API_KEY.
# 2. Run `python build_index.py` once to create the database.
# 3. Run `uvicorn main:app --reload` in your terminal.
# 4. Access the interactive API documentation at http://127.0.0.1:8000/docs