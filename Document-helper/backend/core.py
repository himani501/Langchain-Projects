from typing import Any, Dict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings

load_dotenv()

# Initialize embeddings (same as ingestion.py)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

#Initialize vector store
vectorstore = PineconeVectorStore(
    index_name="medium-blogs-embeddings-768", embedding=embeddings
)
# Initialize chat model - using llama3.2 which supports tools better
model = init_chat_model("llama3.2", model_provider="ollama")


def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain."""
    # Retrieve top 4 most similar documents
    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)
    
    # Serialize documents for the model
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    
    return serialized, retrieved_docs


def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.
    
    Args:
        query: The user's question
        
    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """
    # Retrieve relevant context
    context_text, context_docs = retrieve_context(query)
    
    # Create a prompt with retrieved context
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "Use the provided context to answer the user's question. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the provided context, say so clearly."
    )
    
    user_prompt = f"""
Context:
{context_text}

Question: {query}

Please answer the question based on the provided context. If the context doesn't contain enough information to answer the question, please say so.
"""
    
    # Generate response using the model directly
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response = model.invoke(messages)
    
    return {
        "answer": response.content,
        "context": context_docs
    }

if __name__ == '__main__':
    result = run_llm(query="what are deep agents?")
    print(result)
    