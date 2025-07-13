from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
# from langchain_huggingface.embeddings import 
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from langchain_community.vectorstores import FAISS
from api.utils.config import EXPORT_TYPE, EMBED_MODEL_ID
from langchain_docling.loader import ExportType

# Global vectorstore to persist across requests
vectorstore = None

async def embed_documents(docs: List[Document]) -> FAISS:
    """Embed documents and create FAISS vectorstore"""
    global vectorstore
    
    if not docs:
        raise ValueError("No documents provided for embedding")
    
    # Split docs based on EXPORT_TYPE
    if EXPORT_TYPE == ExportType.DOC_CHUNKS:
        splits = docs
    elif EXPORT_TYPE == ExportType.MARKDOWN:
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header_1"),
                ("##", "Header_2"),
                ("###", "Header_3"),
            ],
        )
        # Split each doc's page_content into smaller chunks
        splits = []
        for doc in docs:
            if doc.page_content:  # Check if page_content exists
                text_splits = splitter.split_text(doc.page_content)
                # Convert text splits back to Document objects
                for split in text_splits:
                    # Ensure split has content before adding
                    if hasattr(split, 'page_content') and split.page_content.strip():
                        splits.append(Document(page_content=split.page_content, metadata=split.metadata))
                    elif isinstance(split, str) and split.strip():
                        # Handle case where split is just a string
                        splits.append(Document(page_content=split, metadata=doc.metadata))
        
        # Fallback to original docs if no splits were created
        if not splits:
            splits = docs
    else:
        raise ValueError(f"Unexpected export type: {EXPORT_TYPE}")

    # Filter out empty documents
    splits = [doc for doc in splits if doc.page_content and doc.page_content.strip()]
    
    if not splits:
        raise ValueError("No valid document content found after processing")

    # Create embeddings
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="nomic-ai/nomic-embed-text-v1",
        model_kwargs={"device": "cpu", "trust_remote_code": True},
        # encode_kwargs={"normalize_embeddings": True},
        # query_instruction="search_query:",
        # embed_instruction="search_document:"
    ) 
    # Extract texts for FAISS
    texts = [chunk.page_content for chunk in splits]
    metadatas = [chunk.metadata for chunk in splits]

    # Create vectorstore
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    
    return vectorstore 

def get_vectorstore():
    """Get the global vectorstore"""
    global vectorstore 
    return vectorstore

def clear_vectorstore():
    """Clear the global vectorstore"""
    global vectorstore
    vectorstore = None