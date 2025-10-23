from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter()

# Request model
class QueryRequest(BaseModel):
    query: str

# Response model
class QueryResponse(BaseModel):
    response: str
    sources: list = []
    used_fallback: bool = False


@router.post("/query", response_model=QueryResponse)
async def query_documents_endpoint(
    query_request: QueryRequest,
    request: Request
):
    """Query the document collection using the new RAG pipeline."""
    try:
        # Access RAG pipeline from app state
        rag_pipeline = request.app.state.rag_pipeline
        
        # Use the new RAG pipeline to get answer
        result = rag_pipeline.answer(query_request.query)
        
        return {
            "response": result["answer"],
            "sources": result["sources"],
            "used_fallback": result["used_fallback"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

