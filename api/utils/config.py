import os
from dotenv import load_dotenv
from langchain_docling.loader import ExportType

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

# Embedding and generation model
EMBED_MODEL_ID = "nomic-ai/nomic-embed-text-v1"

# Generation model
GEN_MODEL_ID = "mistral:latest" 
TEMPERATURE = 0.3

# Ollama model parameters - Increased for complete responses
OLLAMA_NUM_CTX = 16384
OLLAMA_NUM_PREDICT = 4096
OLLAMA_TOP_K = 20
OLLAMA_TOP_P = 0.8
OLLAMA_REPEAT_PENALTY = 1.1

# Export type for Docling loader
EXPORT_TYPE = ExportType.DOC_CHUNKS

# Number of top results to retrieve - Increased for better context
TOP_K = 5  # Increased from 3