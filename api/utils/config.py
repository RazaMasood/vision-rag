import os
from dotenv import load_dotenv
from langchain_docling.loader import ExportType

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

# Embedding and generation model
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Generation model
GEN_MODEL_ID = "mistral"
TEMPERATURE = 0.5

# Export type for Docling loader
EXPORT_TYPE = ExportType.DOC_CHUNKS

# Number of top results to retrieve
TOP_K = 3