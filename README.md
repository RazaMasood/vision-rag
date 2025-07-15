# RAG Document Processing

Upload documents and chat with them using local AI models.

## Features

- Upload PDF, DOCX, TXT, MD, HTML, jpg, png files or URLs
- Ask questions about your documents
- Local processing with Ollama
- Web chat interface

## Quick Start

1. **Install Ollama and pull Mistral model**:
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull mistral:latest
```

2. **Install dependencies**:
```bash
pip install -e .
```

3. **Run the API**:
```bash
python main.py
```

4. **Run the web interface**:
```bash
streamlit run ui.py
```

5. **Open** `http://localhost:8501` in your browser

## API Usage

**Upload document**:
```bash
curl -X POST "http://localhost:8010/upload/" -F "file=@document.pdf"
```

**Ask questions**:
```bash
curl -X POST "http://localhost:8010/query/" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

## Tech Stack

- **Backend**: FastAPI
- **Frontend**: Streamlit  
- **LLM**: Ollama (Mistral)
- **Vector DB**: FAISS
- **Document Processing**: Docling

## Requirements

- Python 3.12+
- Ollama with Mistral model
