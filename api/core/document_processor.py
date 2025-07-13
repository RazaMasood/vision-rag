import asyncio
from typing import List
from langchain_docling.loader import DoclingLoader
from langchain_core.documents import Document
from docling.chunking import HybridChunker

from api.utils.config import EXPORT_TYPE, EMBED_MODEL_ID
from api.core.embedding import embed_documents, clear_vectorstore
import tempfile
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def process_uploaded_file(file) -> List[Document]:
    """Process uploaded file and return documents"""
    logger.info(f"Processing uploaded file: {file.filename}")
    
    # Clear existing vectorstore before processing new file
    clear_vectorstore()
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        # Load document using DoclingLoader
        loader = DoclingLoader(tmp_file_path, export_type=EXPORT_TYPE, chunker=HybridChunker(tokenizer=EMBED_MODEL_ID))
        docs = loader.load()
        
        logger.info(f"Loaded {len(docs)} documents from file")
        
        # Validate documents
        if not docs:
            raise ValueError("No documents were extracted from the file")
        
        # Log document content for debugging
        for i, doc in enumerate(docs[:3]):  # Log first 3 docs
            logger.info(f"Document {i+1}: {len(doc.page_content)} characters")
            if doc.page_content:
                logger.info(f"Sample content: {doc.page_content[:100]}...")
        
        # Store in vectorstore
        vectorstore = await embed_documents(docs)
        logger.info(f"Successfully created vectorstore with {vectorstore.index.ntotal} embeddings")
        
        return docs
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

async def process_uploaded_url(url: str) -> List[Document]:
    """Process URL and return documents"""
    logger.info(f"Processing URL: {url}")
    
    # Clear existing vectorstore before processing new URL
    clear_vectorstore()
    
    try:
        # Load document from URL using DoclingLoader
        loader = DoclingLoader(str(url), export_type=EXPORT_TYPE, chunker=HybridChunker(tokenizer=EMBED_MODEL_ID))
        docs = loader.load()
        
        logger.info(f"Loaded {len(docs)} documents from URL")
        
        # Validate documents
        if not docs:
            raise ValueError("No documents were extracted from the URL")
        
        # Log document content for debugging
        for i, doc in enumerate(docs[:3]):  # Log first 3 docs
            logger.info(f"Document {i+1}: {len(doc.page_content)} characters")
            if doc.page_content:
                logger.info(f"Sample content: {doc.page_content[:100]}...")
        
        # Store in vectorstore
        vectorstore = await embed_documents(docs)
        logger.info(f"Successfully created vectorstore with {vectorstore.index.ntotal} embeddings")
        
        return docs
        
    except Exception as e:
        logger.error(f"Error processing URL: {str(e)}")
        raise