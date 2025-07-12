from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from api.core.embedding import get_vectorstore
from api.utils.config import GEN_MODEL_ID, TOP_K, TEMPERATURE

def clip_text(text: str, threshold: int = 100) -> str:
    """Clip text to specified threshold"""
    return f"{text[:threshold]}..." if len(text) > threshold else text

async def query_documents(question: str) -> dict:
    """Query documents and return answer with sources"""
    vectorstore = get_vectorstore()
    
    if vectorstore is None:
        return {
            "answer": "No documents have been uploaded yet. Please upload a document first.",
            "sources": []
        }
    
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    
    # Create LLM
    llm = OllamaLLM(model=GEN_MODEL_ID, temperature=TEMPERATURE)
    
    # Create prompt template
    prompt = PromptTemplate(
        input_variables=["context", "input"],
        template="""
You are a helpful assistant.
Answer ONLY from the provided Document context.
If the context is insufficient, just say: "This topic is not discussed in the document."

### CONTEXT FROM DOCUMENT:
{context}

### IMPORTANT INSTRUCTIONS:
- Do NOT guess or use outside knowledge.
- Be concise.

### Question: {input}

### Answer:
"""
    )
    
    # Create chains
    try:
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        # Get response
        resp_dict = rag_chain.invoke({"input": question})
        
        # Format sources
        sources = []
        for i, doc in enumerate(resp_dict.get("context", [])):
            source_info = {
                "source_id": i + 1,
                "text": clip_text(doc.page_content, threshold=350),
                "metadata": {}
            }
            
            # Add metadata (excluding 'pk' key)
            for key, val in doc.metadata.items():
                if key != "pk":
                    clipped_val = clip_text(str(val)) if isinstance(val, str) else val
                    source_info["metadata"][key] = clipped_val
            
            sources.append(source_info)
        
        return {
            "answer": clip_text(resp_dict.get("answer", "No answer found."), threshold=350),
            "sources": sources
        }
    
    except Exception as e:
        return {
            "answer": f"Error processing query: {str(e)}",
            "sources": []
        }