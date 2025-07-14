import streamlit as st
import requests
import json
from typing import Optional
import time
from urllib.parse import urlparse

# Configuration
API_BASE_URL = "http://localhost:8010"
UPLOAD_ENDPOINT = f"{API_BASE_URL}/upload/"
QUERY_ENDPOINT = f"{API_BASE_URL}/query/"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"

# Supported file types
SUPPORTED_EXTENSIONS = ['pdf', 'docx', 'txt', 'md', 'html']

def check_api_health() -> bool:
    """Check if the API is running"""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        return response.status_code == 200
    except:
        return False

def is_valid_url(url: str) -> bool:
    """Validate URL format"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def upload_file(file) -> dict:
    """Upload file to API"""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        response = requests.post(UPLOAD_ENDPOINT, files=files, timeout=120)
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": response.text}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Connection error: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"Upload error: {str(e)}"}

def upload_url(url: str) -> dict:
    """Upload URL to API"""
    try:
        headers = {"X-Document-URL": url}
        response = requests.post(UPLOAD_ENDPOINT, headers=headers, timeout=120)
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": response.text}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Connection error: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"URL processing error: {str(e)}"}

def query_documents(question: str) -> dict:
    """Query documents via API"""
    try:
        payload = {"question": question}
        response = requests.post(QUERY_ENDPOINT, json=payload, timeout=120)
        
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": response.text}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": f"Connection error: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": f"Query error: {str(e)}"}

def main():
    st.set_page_config(page_title="Document Chat", page_icon="💬")
    
    # Initialize session state
    if 'document_uploaded' not in st.session_state:
        st.session_state.document_uploaded = False
    if 'document_info' not in st.session_state:
        st.session_state.document_info = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Simple header
    st.title("Document Chat")
    
    # Check API health
    if not check_api_health():
        st.error("API server not running on http://localhost:8010")
        st.stop()
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("Upload Document")
        
        # Upload method selection
        upload_method = st.radio("Choose upload method:", ["File", "URL"])
        
        if upload_method == "File":
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=[ext[1:] for ext in SUPPORTED_EXTENSIONS]
            )
            
            if uploaded_file:
                st.write(f"Selected: {uploaded_file.name}")
                
                if st.button("Upload"):
                    with st.spinner("Uploading..."):
                        result = upload_file(uploaded_file)
                        
                        if result["success"]:
                            st.session_state.document_uploaded = True
                            st.session_state.document_info = result["data"]
                            st.success("File uploaded successfully!")
                            st.rerun()
                        else:
                            st.error(f"Upload failed: {result['error']}")
        
        else:  # URL upload
            url_input = st.text_input("Enter document URL:")
            
            if url_input:
                if is_valid_url(url_input):
                    if st.button("Upload"):
                        with st.spinner("Processing URL..."):
                            result = upload_url(url_input)
                            
                            if result["success"]:
                                st.session_state.document_uploaded = True
                                st.session_state.document_info = result["data"]
                                st.success("URL processed successfully!")
                                st.rerun()
                            else:
                                st.error(f"Upload failed: {result['error']}")
                else:
                    st.warning("Please enter a valid URL")
        
        # Document status
        if st.session_state.document_uploaded:
            st.success(f"Document ready: {st.session_state.document_info['source']}")
            
            if st.button("Remove Document"):
                st.session_state.document_uploaded = False
                st.session_state.document_info = None
                st.session_state.chat_history = []
                st.rerun()
    
    # Chat section
    if st.session_state.document_uploaded:
        st.header("Chat")
        
        # Display chat history
        for question, answer in st.session_state.chat_history:
            st.write("👤 ", question)
            st.write("🤖 ", answer)
            st.write("---")
        
        # Chat input
        question = st.text_input("Ask a question about your document:")
        
        if st.button("Send") and question.strip():
            with st.spinner("Getting answer..."):
                result = query_documents(question)
                
                if result["success"]:
                    st.session_state.chat_history.append((question, result["data"]["answer"]))
                    st.rerun()
                else:
                    st.error(f"Error: {result['error']}")
    
    else:
        st.info("Upload a document to start chatting")

if __name__ == "__main__":
    main()