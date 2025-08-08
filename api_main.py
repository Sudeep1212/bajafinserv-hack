
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import List
import logging
import time
import sys
import os
import sys

# This line ensures that the current directory is in the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the new system from core_logic
from core_logic import hackrx_system

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="HackRx 6.0 - A6 Memory-Optimized RAG System",
    description="A memory-efficient system using BM25 and Gemini for document Q&A.",
    version="6.0.0"
)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# Authentication
async def verify_token(authorization: str = Header(None)):
    """Verify the Bearer token."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    # This is the required token for the HackRx API
    expected_token = "0fce51ab380da7e61785e46ae2ba8cee5037bae3ff8d86c68b1a4a1cefe03556"
    if authorization != f"Bearer {expected_token}":
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return True

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "HackRx 6.0 - A6 Memory-Optimized RAG System is running.",
        "status": "healthy",
        "version": "6.0.0"
    }

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_hackrx_system_endpoint(
    request: QueryRequest,
    token_verified: bool = Depends(verify_token)
):
    """
    Run the memory-optimized HackRx 6.0 system.
    """
    start_time = time.time()
    logger.info("=== Received new request for /hackrx/run ===")
    
    try:
        if not request.documents:
            raise HTTPException(status_code=400, detail="Document URL is required")
        if not request.questions:
            raise HTTPException(status_code=400, detail="At least one question is required")
        
        logger.info(f"Processing {len(request.questions)} questions from document: {request.documents}")
        
        # Process questions using the new system
        result = hackrx_system.process_questions(request.documents, request.questions)
        
        total_time = time.time() - start_time
        logger.info(f"=== Completed request in {total_time:.2f}s ===")
        
        return QueryResponse(answers=result['answers'])
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)