from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from api.utils.config import EXPORT_TYPE, EMBED_MODEL_ID
from langchain_docling.loader import ExportType

async def embed_documents(docs: List[Document]) -> FAISS:
    """Embed documents and create FAISS vectorstore"""
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
            text_splits = splitter.split_text(doc.page_content)
            # Convert text splits back to Document objects
            for split in text_splits:
                splits.append(Document(page_content=split, metadata=doc.metadata))
    else:
        raise ValueError(f"Unexpected export type: {EXPORT_TYPE}")

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)

    # Extract texts for FAISS
    texts = [chunk.page_content for chunk in splits]
    metadatas = [chunk.metadata for chunk in splits]

    # Create vectorstore
    vectorstore   = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    
    return vectorstore 

def get_vectorstore():
    global vectorstore 
    return vectorstore