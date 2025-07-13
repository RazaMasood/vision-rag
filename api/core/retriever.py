from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from api.core.embedding import get_vectorstore
from api.utils.config import (GEN_MODEL_ID, TOP_K, TEMPERATURE, OLLAMA_NUM_CTX, OLLAMA_NUM_PREDICT, OLLAMA_REPEAT_PENALTY, OLLAMA_TOP_K, OLLAMA_TOP_P)
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clip_text(text: str, threshold: int = 5000) -> str:
    """Clip text to specified threshold"""
    if not text:
        return ""
    return f"{text[:threshold]}..." if len(text) > threshold else text

async def query_documents(question: str) -> dict:
    """Query documents and return answer with sources"""
    try:
        # Input validation
        if not question or not question.strip():
            return {
                "answer": "Please provide a valid question.",
                "sources": []
            }
        
        vectorstore = get_vectorstore()
        
        if vectorstore is None:
            return {
                "answer": "No documents have been uploaded yet. Please upload a document first.",
                "sources": []
            }
        
        # Create retriever with error handling
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
            
            # Test retrieval with a simple search
            test_docs = retriever.invoke(question)
            if not test_docs:
                return {
                    "answer": "No relevant information found in the uploaded documents.",
                    "sources": []
                }
            
        except Exception as e:
            logger.error(f"Error creating retriever: {str(e)}")
            return {
                "answer": f"Error accessing document index: {str(e)}",
                "sources": []
            }
        
        # Create LLM with error handling
        try:
            # Create LLM with increased context window and higher token limit
            llm = OllamaLLM(
                model=GEN_MODEL_ID, 
                temperature=TEMPERATURE,
                num_ctx=OLLAMA_NUM_CTX,
                num_predict=4096,  # Increased from 2048 to allow longer responses
                top_k=OLLAMA_TOP_K,
                top_p=OLLAMA_TOP_P,
                repeat_penalty=OLLAMA_REPEAT_PENALTY,
            )

        except Exception as e:
            logger.error(f"Error creating LLM: {str(e)}")
            return {
                "answer": f"Error initializing language model: {str(e)}. Please ensure Ollama is running and the model '{GEN_MODEL_ID}' is available.",
                "sources": []
            }
        
        # Create prompt template with instruction to provide complete answers
        prompt = PromptTemplate(
            input_variables=["context", "input"],
            template="""<s>[INST] You are a comprehensive document assistant capable of data extraction, calculations, KPI analysis, and summarization.

DOCUMENT:
{context}

RULES:
1. Use ONLY information from the document above
2. If information is missing, say: "Not available in document"
3. Be accurate and complete - provide FULL detailed answers
4. Quote directly when helpful
5. Show all calculation steps and reasoning
6. Don't truncate your response - provide the complete answer

TASK TYPES:

Data Extraction:
- Copy exact values, dates, names
- Note if text is unclear

Calculations:
- Use only numbers from the document
- Show your work and source: "Calculation: [your formula] = [result], using values from document"
- State if required data is missing
- Provide complete step-by-step calculations

KPI Analysis:
- Create appropriate formulas using document data
- Show calculation steps and data sources
- Explain the KPI meaning if relevant

Summaries:
- Include key points and data
- Maintain document context
- Provide comprehensive summaries

QUESTION: {input}

Provide a complete, detailed answer based solely on the document: [/INST]"""
        )
        
        # Create chains with error handling
        try:
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            
            # Get response
            resp_dict = rag_chain.invoke({"input": question})
            
            # Validate response
            if not resp_dict:
                return {
                    "answer": "Error: No response generated.",
                    "sources": []
                }
            
            # Format sources
            sources = []
            context_docs = resp_dict.get("context", [])
            
            for i, doc in enumerate(context_docs):
                if not doc or not hasattr(doc, 'page_content'):
                    continue
                    
                source_info = {
                    "source_id": i + 1,
                    "text": clip_text(doc.page_content, threshold=500),  # Increased source text limit
                    "metadata": {}
                }
                
                # Add metadata (excluding 'pk' key)
                if hasattr(doc, 'metadata') and doc.metadata:
                    for key, val in doc.metadata.items():
                        if key != "pk":
                            clipped_val = clip_text(str(val)) if isinstance(val, str) else val
                            source_info["metadata"][key] = clipped_val
                
                sources.append(source_info)
            
            answer = resp_dict.get("answer", "No answer found.")
            
            # Return complete answer without clipping
            return {
                "answer": answer,  
                "sources": sources  # Include sources for reference
            }
        
        except Exception as e:
            logger.error(f"Error in RAG chain: {str(e)}")
            return {
                "answer": f"Error processing query with language model: {str(e)}",
                "sources": []
            }
    
    except Exception as e:
        logger.error(f"Unexpected error in query_documents: {str(e)}")
        return {
            "answer": f"Unexpected error processing query: {str(e)}",
            "sources": []
        }