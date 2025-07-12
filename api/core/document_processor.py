import asyncio
from typing import List
from langchain_docling.loader import DoclingLoader
from langchain_core.documents import Document
from api.utils.config import EXPORT_TYPE
from api.core.embedding import embed_documents
import tempfile
import os

# Global vectorstore to persist across requests
vectorstore = None

async def process_uploaded_file(file) -> List[Document]:
    """Process uploaded file and return documents"""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        # Load document using DoclingLoader
        loader = DoclingLoader(tmp_file_path, export_type=EXPORT_TYPE)
        docs = loader.load()
        
        # Store in global vectorstore
        global vectorstore
        vectorstore = await embed_documents(docs)
        
        return docs
    finally:
        # Clean up temporary file
        os.unlink(tmp_file_path)

async def process_uploaded_url(url: str) -> List[Document]:
    """Process URL and return documents"""
    # Load document from URL using DoclingLoader
    loader = DoclingLoader(str(url), export_type=EXPORT_TYPE)
    docs = loader.load()
    
    # Store in global vectorstore
    global vectorstore
    vectorstore = await embed_documents(docs)
    
    return docs