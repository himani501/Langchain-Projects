# LangChain Projects

A collection of AI projects built with LangChain, demonstrating progressively advanced patterns from basic prompts to RAG pipelines to agentic tool calling.

## Project Structure

```
Langchain-Projects/
├── main.py                    # Hello World - basic LLM summarization
├── tools/                     # Search tool implementations
│   ├── tavily_search.py       # Tavily search agent with structured output
│   └── tools_search.py        # Custom search tools
├── RAG/                       # Retrieval-Augmented Generation
│   ├── ingestion.py           # Document ingestion to Pinecone
│   ├── RAG.py                 # RAG pipeline (LCEL vs non-LCEL)
│   └── mediumblog.txt         # Sample data
├── Ecommerce-Agent/           # Agent loop implementations
│   ├── AgentLoop_Raw_Function_calling.py
│   └── AgentLoop_using_langchain_tool_calling.py
├── Document-helper/           # Streamlit RAG chat app
│   ├── main.py                # Streamlit UI
│   ├── ingestion.py           # Doc ingestion
│   └── backend/
│       └── core.py            # RAG pipeline logic
└── Agent/                     # Placeholder for future work
```

## Projects

### 1. Hello World (`main.py`)

Basic LangChain example demonstrating prompt templates and LLM invocation.

```python
# Uses Ollama with gemma3 model
llm = ChatOllama(temperature=0, model="gemma3:270m")
chain = summary_prompt_template | llm
result = chain.invoke({"information": information})
```

**Run:** `uv run python main.py`

---

### 2. Search Agent (`tools/tavily_search.py`)

Modern search agent using LangChain's `create_agent` interface with Tavily integration and Pydantic structured outputs.

**Features:**
- Tavily web search integration
- Structured response schema with `BaseModel`
- Returns answer + source URLs

```python
class AgentResponse(BaseModel):
    answer: str
    sources: List[Source]

agent = create_agent(model=llm, tools=[TavilySearch()], response_format=AgentResponse)
```

---

### 3. RAG Pipeline (`RAG/`)

Demonstrates Retrieval-Augmented Generation with two implementation styles.

#### Ingestion (`RAG/ingestion.py`)
- Loads text documents with `TextLoader`
- Splits into chunks (1000 chars) with `CharacterTextSplitter`
- Embeds with `nomic-embed-text` (768 dimensions)
- Stores in Pinecone serverless index

```bash
uv run python RAG/ingestion.py
```

#### Retrieval (`RAG/RAG.py`)

**Without LCEL** - Manual step-by-step:
```python
docs = retriever.invoke(query)
context = format_docs(docs)
messages = prompt_template.format_messages(context=context, question=query)
response = llm.invoke(messages)
```

**With LCEL** - Declarative chain (recommended):
```python
retrieval_chain = (
    RunnablePassthrough.assign(
        context=itemgetter("question") | retriever | format_docs
    )
    | prompt_template
    | llm
    | StrOutputParser()
)
```

LCEL advantages: streaming, async, batch processing, composability.

```bash
uv run python RAG/RAG.py
```

---

### 4. Ecommerce Agent (`Ecommerce-Agent/`)

Shopping assistant demonstrating agent loops with tool calling.

**Tools:**
- `get_product_price(product)` - Look up catalog prices
- `apply_discount(price, tier)` - Apply bronze/silver/gold discounts

**Features:**
- Custom `@tool` decorators
- Agent loop with max iterations
- LangSmith tracing with `@traceable`
- Error handling for tool failures

```python
@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    prices = {"laptop": 1299.99, "headphones": 89.00, "keyboard": "79.65"}
    return prices.get(product, 0)

llm_with_tools = llm.bind_tools(tools)
```

```bash
uv run python Ecommerce-Agent/AgentLoop_using_langchain_tool_calling.py
```

---

### 5. Document Helper (`Document-helper/`)

Streamlit chat application for querying LangChain documentation with source citations.

**Features:**
- Chat interface with message history
- RAG-powered responses
- Expandable source citations
- Session state management
- Clear chat functionality

**Architecture:**
- `main.py` - Streamlit UI
- `backend/core.py` - RAG pipeline with context retrieval
- Uses Pinecone index `medium-blogs-embeddings-768`

```bash
streamlit run Document-helper/main.py
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| **LLMs** | Ollama (llama3.2, gemma3) |
| **Vector DB** | Pinecone (serverless, AWS us-east-1) |
| **Embeddings** | nomic-embed-text (768 dims) |
| **Framework** | LangChain, LangSmith |
| **Web Search** | Tavily |
| **UI** | Streamlit |
| **Package Manager** | uv |

## Setup

### 1. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Or with uv:
```bash
uv venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Configure environment variables

Create a `.env` file:

```env
PINECONE_API_KEY=your_pinecone_api_key
INDEX_NAME=medium-blogs-embeddings-768
TAVILY_API_KEY=your_tavily_api_key
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
```

### 4. Install Ollama models

```bash
ollama pull llama3.2
ollama pull gemma3:270m
ollama pull nomic-embed-text
```

## Dependencies

Key packages from `pyproject.toml`:

- `langchain>=1.2.10`
- `langchain-ollama>=1.0.1`
- `langchain-pinecone>=0.2.13`
- `langchain-tavily>=0.2.17`
- `langsmith>=0.7.4`
- `streamlit>=1.54.0`
- `pydantic>=2.12.5`

## Quick Start

```bash
# Clone and setup
cd Langchain-Projects
uv sync

# Run hello world
uv run python main.py

# Run RAG demo
uv run python RAG/ingestion.py  # First, ingest data
uv run python RAG/RAG.py        # Then, query

# Run agent demo
uv run python Ecommerce-Agent/AgentLoop_using_langchain_tool_calling.py

# Run Streamlit app
streamlit run Document-helper/main.py
```

## Learning Path

1. **Start here:** `main.py` - Basic prompt templates
2. **Add tools:** `tools/tavily_search.py` - External tool integration
3. **Add retrieval:** `RAG/RAG.py` - Vector search + context injection
4. **Add agents:** `Ecommerce-Agent/` - Tool calling loops
5. **Build apps:** `Document-helper/` - Production-ready Streamlit UI
