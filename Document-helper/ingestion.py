import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
# from langchain_chroma import Chroma

load_dotenv()

#configure SSL
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)
vectorStore = PineconeVectorStore(index_name="medium-blogs-embeddings-768", embedding=embeddings)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()

async def index_documents_async(documents: List[Document], batch_size: int = 50):
    """Process documents in batch asynchronously"""
    print("VECTOR STORAGE PHASE")

    # Create batches
    batches = [
        documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
    ]
    print(f"Vectorstore Indexing: Split into {len(batches)} batches of batchSize {batch_size} each")

    # process all documents concurrently
    async def add_batch(batch: List[Document], batch_num: int):
        try:
            await vectorStore.aadd_documents(batch)
            print("Successfully added batch")
        except Exception as e:
            print('Failed to add batch')
            return False
        return True
    
    # process batches concurrently
    tasks = [add_batch(batch, i+1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successful batches
    successful = sum (1 for result in results if result is True)
    if successful == len(batches):
        print("All batches processed successfully!")
    else:
        print("Error") 


async def main():
    """Main Async function to orchestrate the entire process"""

    # Crawl the documentation site

    res = tavily_crawl.invoke({
        "url": "https://python.langchain.com",
        "max_depth": 5, # it will crawl upto 5 depth pages
        "extract_depth": "advanced",
        "instructions": "content on ai agents"  #it will fetch pages related to ai agents only
    })

    all_docs = [Document(page_content=result['raw_content'], metadata={"source": result['url']}) for result in res['results']]
    print(f"Successfully crawled {len(all_docs)} URLs from documentation site")

    # Split documents into chunks 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_docs)
    print(f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_docs)} documents")

    # Process documents asynchronously
    await index_documents_async(splitted_docs, batch_size=500)

if __name__ == "__main__":
    asyncio.run(main())