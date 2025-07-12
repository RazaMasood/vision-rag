from fastapi import APIRouter, UploadFile, File, Header, HTTPException
from typing import Optional
from api.schemas.document_processor import URLUploadRequest
from api.core.document_processor import process_uploaded_file, process_uploaded_url

router = APIRouter(tags=["upload"])

@router.post("/upload/",
             summary="Upload and process a document",
             description="Upload a file or a URL to process. Only one method should be used at a time.")
async def upload_document(
    file: Optional[UploadFile] = File(
        None, 
        description="Document file to upload and process",
        example="document.pdf"
    ),
    url: Optional[str] = Header(None, alias="X-Document-URL")
) -> dict:

    if not file and not url:
        raise HTTPException(status_code=400, detail="Either file or URL must be provided.")
    if file and url:
        raise HTTPException(status_code=400, detail="Provide either a file or a URL, not both.")

    if file:
        docs = await process_uploaded_file(file)
        return {
            "status": "processed",
            "source": file.filename,
            "document_count": len(docs)
        }

    try:
        valid_url = URLUploadRequest(url=url)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid URL format.")

    docs = await process_uploaded_url(valid_url.url)
    return {
        "status": "processed",
        "source": str(valid_url.url),
        "document_count": len(docs)
    }