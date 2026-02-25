import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

if __name__ == '__main__':
    print("Ingesting....")
    loader = TextLoader('/Users/himanibhardwaj/himcodes/GenAI/Langchain-Projects/RAG/mediumblog.txt', encoding='UTF-8')
    document = loader.load()    # load the file in langchain document

    print("splitting....")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f'Created {len(texts)} chunks')

    embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
    
    # Initialize Pinecone and create index with correct dimensions
    pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
    index_name = "medium-blogs-embeddings-768"
    
    # Check if index exists, if not create it
    if index_name not in pc.list_indexes().names():
        print(f"Creating new index: {index_name}")
        pc.create_index(
            name=index_name,
            dimension=768,  # nomic-embed-text produces 768-dimensional embeddings
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    else:
        print(f"Using existing index: {index_name}")

    print("ingesting....")
    PineconeVectorStore.from_documents(texts, embeddings_model, index_name=index_name)
    print("finish")