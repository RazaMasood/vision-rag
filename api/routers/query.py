from fastapi import APIRouter, HTTPException
from api.schemas.document_processor import QueryRequest, QueryResponse
from api.core.retriever import query_documents

router = APIRouter(tags=["query"])

@router.post("/query/",
             summary="Query uploaded documents",
             description="Ask questions about uploaded documents",
             response_model=QueryResponse)
async def query_document(request: QueryRequest) -> QueryResponse:
    """Query the uploaded documents with a question"""
    try:
        result = await query_documents(request.question)
        return QueryResponse(
            answer=result["answer"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")