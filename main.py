# main.py - Production Ready Version with Bug Fixes
# This script runs a FastAPI web server to handle user queries.
# It loads the pre-built vector database, connects to the Groq API for LLM reasoning,
# and uses LangChain to orchestrate the Retrieval-Augmented Generation (RAG) pipeline.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from dotenv import load_dotenv
import os
import logging
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from the .env file.
load_dotenv()

# --- Pydantic Models for API Input and Output ---
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Insurance claim query")

class Justification(BaseModel):
    decision: str = Field(description="The final decision, either 'Approved' or 'Rejected'.")
    amount: float = Field(description="The payable amount. Should be 0.0 if rejected.")
    reason: str = Field(description="A clear, concise justification for the decision based on policy clauses.")
    clause_references: list[str] = Field(description="A list of specific clauses and their source documents that support the decision.")

class ErrorResponse(BaseModel):
    error: str
    message: str

# --- FastAPI App Initialization ---
app = FastAPI(
    title="HackRx 6.0 LLM Document Processor (Production API)",
    description="A production-ready API-powered system for processing insurance policy documents using RAG.",
    version="2.0.0"
)

# --- Global Component Loading ---
try:
    logger.info("Loading embedding model and vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    logger.info("Components loaded successfully.")
except Exception as e:
    logger.error(f"Error loading components: {e}")
    retriever = None

# Initialize the Groq LLM with error handling
try:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    
    llm = ChatGroq(
        model_name="llama3-70b-8192",
        temperature=0,
        api_key=api_key,
        max_retries=2
    )
    logger.info("LLM initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing LLM: {e}")
    llm = None

# --- Enhanced Prompt Engineering ---
parser_prompt_template = """
You are an expert at parsing insurance queries. Extract key information and structure it clearly.

Query: {query}

Please extract:
- Procedure/Treatment type
- Cost amount (if mentioned)
- Policy tenure/duration
- Any other relevant details

Return a clear, structured summary:
"""

# Improved final prompt with better calculation instructions
final_prompt_template = """
You are an expert insurance claims adjudicator with strong mathematical skills. 
Analyze this claim based ONLY on the provided policy clauses.

**User's Query Details:**
{structured_query}

**Relevant Policy Clauses:**
---
{context}
---

**CRITICAL INSTRUCTIONS:**
1. Check waiting periods carefully (30-day, 90-day, 12-month rules)
2. Apply sub-limits exactly (e.g., cataract: 25,000 per eye)
3. Calculate co-payments precisely (e.g., dental: 20% deduction)
4. For multiple procedures, calculate each separately then sum
5. Never exceed the stated sub-limits
6. If cost > sub-limit, payable amount = sub-limit

**MATHEMATICAL EXAMPLES:**
- Single cataract (cost 30k): Pay 25k (sub-limit)
- Dual cataract (cost 60k): Pay 50k (25k × 2 eyes)
- Dental (cost 8k, 20% co-pay): Pay 6.4k (8k × 0.8)

**CRITICAL: Return ONLY valid JSON, nothing else. No explanations, no text before or after.**

**REQUIRED JSON FORMAT:**
{{
  "decision": "Approved" or "Rejected",
  "amount": [exact_calculated_number],
  "reason": "Step-by-step calculation and policy reasoning",
  "clause_references": ["specific clause citations"]
}}

**IMPORTANT: Your response must be ONLY the JSON object above. Do not include any other text.**
"""

PARSER_PROMPT = PromptTemplate.from_template(parser_prompt_template)
FINAL_PROMPT = PromptTemplate.from_template(final_prompt_template)

# --- Chains ---
structuring_chain = PARSER_PROMPT | llm | StrOutputParser()

def process_query_pipeline(query: str):
    """Enhanced pipeline with error handling and validation"""
    
    try:
        # Input validation
        if not query or len(query.strip()) < 3:
            raise ValueError("Query too short or empty")
        
        logger.info(f"Processing query: {query[:100]}...")
        
        # Step 1: Structure the query
        try:
            structured_query = structuring_chain.invoke(query)
            logger.info("Query structured successfully")
        except Exception as e:
            logger.error(f"Error in structuring: {e}")
            structured_query = f"Raw query: {query}"
        
        # Step 2: Retrieve relevant context
        try:
            context_docs = retriever.invoke(query)
            logger.info(f"Retrieved {len(context_docs)} documents")
            
            if not context_docs:
                raise ValueError("No relevant policy documents found")
                
        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            raise ValueError("Unable to retrieve relevant policy information")
        
        # Step 3: Format context
        context = "\n\n".join([doc.page_content for doc in context_docs])
        
        if len(context.strip()) < 10:
            raise ValueError("Retrieved context too short")
            
        # Step 4: Create final prompt
        prompt_input = {
            "structured_query": structured_query,
            "context": context[:8000]  # Limit context length
        }
        
        # Step 5: Get LLM response with retry logic and better JSON extraction
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response = llm.invoke(FINAL_PROMPT.format(**prompt_input))
                response_text = response.content if hasattr(response, 'content') else str(response)
                
                # Extract JSON from response - handle cases where LLM adds extra text
                import re
                import json
                
                # Try to find JSON object in the response
                json_match = re.search(r'\{[^}]*"decision"[^}]*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    try:
                        final_response = json.loads(json_str)
                    except json.JSONDecodeError:
                        # If that fails, try to extract the full object more carefully
                        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                        if json_match:
                            json_str = json_match.group()
                            final_response = json.loads(json_str)
                        else:
                            raise ValueError("No valid JSON found in response")
                else:
                    # Fallback: try the original parser
                    final_response = JsonOutputParser(pydantic_object=Justification).parse(response_text)
                
                # Validate and clean the response
                if not isinstance(final_response, dict):
                    raise ValueError("Invalid response format")
                    
                # Ensure required fields exist and are correct types
                final_response["decision"] = str(final_response.get("decision", "Rejected"))
                final_response["amount"] = float(final_response.get("amount", 0.0))
                final_response["reason"] = str(final_response.get("reason", "Processing error"))
                final_response["clause_references"] = list(final_response.get("clause_references", []))
                
                # Validate decision
                if final_response["decision"] not in ["Approved", "Rejected"]:
                    final_response["decision"] = "Rejected"
                    final_response["amount"] = 0.0
                
                logger.info("Response generated successfully")
                return final_response
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_attempts - 1:
                    raise e
                
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        # Return safe fallback response
        return {
            "decision": "Rejected", 
            "amount": 0.0,
            "reason": f"Unable to process claim due to system error: {str(e)}. Please contact support.",
            "clause_references": []
        }

# --- API Endpoints ---
@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Insurance Claims API is running", "version": "2.0.0"}

@app.get("/health")
async def health_check():
    """Detailed health check"""
    status = {
        "api": "healthy",
        "vector_store": "healthy" if retriever else "error",
        "llm": "healthy" if llm else "error"
    }
    
    if not retriever or not llm:
        raise HTTPException(status_code=503, detail="Service dependencies not available")
    
    return status

@app.post("/process-query", response_model=Justification)
async def process_query(request: QueryRequest):
    """
    Process insurance claim queries with enhanced error handling
    """
    
    # Check system availability
    if not retriever:
        raise HTTPException(
            status_code=503, 
            detail="Vector store not initialized. Please run build_index.py first."
        )
    
    if not llm:
        raise HTTPException(
            status_code=503,
            detail="LLM service not available. Check GROQ_API_KEY configuration."
        )
    
    try:
        # Process the query
        result = process_query_pipeline(request.query)
        
        # Final validation
        if not result or not isinstance(result, dict):
            raise HTTPException(status_code=500, detail="Invalid response generated")
            
        return result
        
    except ValueError as e:
        # Client errors (bad input)
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        # Server errors
        logger.error(f"Unexpected error processing query: {e}")
        raise HTTPException(
            status_code=500, 
            detail="An unexpected error occurred. Please try again or contact support."
        )

# --- Error Handlers ---
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "message": "Please check the API documentation"}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "message": "Please try again later or contact support"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

