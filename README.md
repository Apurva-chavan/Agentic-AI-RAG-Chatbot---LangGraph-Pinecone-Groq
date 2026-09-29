# 🤖 Agentic AI RAG Chatbot

## 🌐 Live Demo

- **API (FastAPI):** https://agentic-ai-rag-chatbot-langgraph.onrender.com
- **Interactive Docs:** https://agentic-ai-rag-chatbot-langgraph.onrender.com/docs
- **GitHub:** https://github.com/Apurva-chavan/Agentic-AI-RAG-Chatbot---LangGraph-Pinecone-Groq

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot built with **LangGraph**, **Pinecone**, and **OpenAI** that answers queries strictly grounded in the *Agentic AI* eBook.

---

## 🏗️ Architecture Overview

```
PDF (Agentic AI eBook)
        │
        ▼
┌──────────────────────┐
│  Ingestion Module    │  PyPDFLoader → RecursiveCharacterTextSplitter
│  src/ingestion.py    │  chunk_size=1000, overlap=200
└────────┬─────────────┘
         │ OpenAI text-embedding-3-small
         ▼
┌──────────────────────┐
│   Pinecone Index     │  Vectors + metadata (text, page number)
│  (agentic-ai-index)  │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────────────────────────────┐
│           LangGraph RAG Workflow             │
│  ┌──────────┐              ┌───────────┐     │
│  │ Retrieve │─────────────▶│ Generate  │     │
│  │  Node    │              │   Node    │     │
│  └──────────┘              └───────────┘     │
│  Pinecone top-k            GPT-4o-mini        │
│  semantic search           grounded answer    │
└──────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────┐
│  FastAPI REST API    │  POST /chat → structured JSON payload
│  (app.py)            │
│  Streamlit UI        │  streamlit_app.py (optional)
└──────────────────────┘
```

### Component Breakdown

| Component | Technology | Responsibility |
|-----------|-----------|----------------|
| Data Ingestion | PyPDFLoader + RecursiveCharacterTextSplitter | Parse PDF, chunk text (1000 chars, 200 overlap) |
| Embeddings & Vector DB | OpenAI `text-embedding-3-small` + Pinecone | Dense vector indexing via LangChain wrapper |
| Orchestration Graph | LangGraph `StateGraph` | START → retrieve → generate → END |
| Confidence Scoring | Context availability heuristic | 0.95 if context retrieved, 0.0 if empty |
| API | FastAPI | `POST /chat` returning answer, context, score |
| UI | Streamlit | Optional interactive chat interface |

---

## 📁 Project Structure

```
.
├── data/
│   └── Ebook-Agentic-AI.pdf        # Downloaded source document (add manually)
├── src/
│   ├── __init__.py
│   ├── config.py                   # Environment setup & constants
│   ├── ingestion.py                # PDF loading, splitting & Pinecone upsert
│   └── graph.py                    # LangGraph workflow definition & state logic
├── app.py                          # FastAPI application (primary interface)
├── streamlit_app.py                # Streamlit UI (optional alternative)
├── tests_sample_queries.py         # 6 sample benchmark queries
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚙️ Setup Guide

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/agentic-ai-rag-chatbot.git
cd agentic-ai-rag-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=agentic-ai-index
PDF_PATH=data/Ebook-Agentic-AI.pdf
```

### 5. Download the eBook PDF

Download the Agentic AI eBook from the provided Google Drive link and place it at:

```
data/Ebook-Agentic-AI.pdf
```

### 6. Run the ingestion pipeline

```bash
python src/ingestion.py
```

This will:
- Load and parse the eBook PDF
- Split into ~1000-character chunks with 200-character overlap
- Generate OpenAI embeddings (`text-embedding-3-small`)
- Create a Pinecone serverless index and upsert all vectors with metadata

### 7. Launch the FastAPI server

```bash
python app.py
```

API available at [http://localhost:8000](http://localhost:8000).  
Interactive docs at [http://localhost:8000/docs](http://localhost:8000/docs).

### 7b. (Alternative) Launch the Streamlit UI

```bash
streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📡 API Usage

### POST `/chat`

**Request:**
```json
{
  "query": "What is Agentic AI?"
}
```

**Response:**
```json
{
  "query": "What is Agentic AI?",
  "final_answer": "Agentic AI refers to autonomous systems that...",
  "retrieved_context_chunks": [
    "Chunk 1 text from PDF...",
    "Chunk 2 text from PDF..."
  ],
  "confidence_score": 0.95
}
```

---

## 🧪 Sample Validation Queries

Run all 6 benchmark queries at once:

```bash
python tests_sample_queries.py
```

| # | Query | Expected Behaviour |
|---|-------|--------------------|
| 1 | What is Agentic AI according to the eBook? | Grounded answer from eBook |
| 2 | How do AI agents differ from traditional automation systems? | Grounded answer from eBook |
| 3 | What are the core components of an Agentic Architecture? | Grounded answer from eBook |
| 4 | What role does memory play in Agentic AI workflows? | Grounded answer from eBook |
| 5 | What key challenges or limitations of Agentic AI are mentioned? | Grounded answer from eBook |
| 6 | Who won the 2022 FIFA World Cup? | ❌ Refused — out-of-scope groundedness check |

---

## 🔑 Key Design Decisions

- **Strict grounding prompt:** LLM is explicitly instructed to answer only from retrieved context; refuses with a clear message if context is insufficient.
- **Confidence score:** `0.95` when context chunks are retrieved, `0.0` when Pinecone returns no results — surfaced in every response payload.
- **LangGraph state machine:** Clean `START → retrieve → generate → END` flow with typed `AgentState`.
- **LangChain PineconeVectorStore:** Used as the vector store wrapper for both ingestion and retrieval, keeping the codebase idiomatic.

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `langgraph` | Graph-based RAG orchestration |
| `langchain` / `langchain-openai` | LLM chains, embeddings, text splitting |
| `langchain-pinecone` | Pinecone vector store integration |
| `langchain-community` | PyPDFLoader |
| `pinecone-client` | Pinecone index management |
| `openai` | GPT-4o-mini + text-embedding-3-small |
| `pypdf` | PDF text extraction backend |
| `streamlit` | Optional web UI |
| `fastapi` + `uvicorn` | REST API server |
| `python-dotenv` | Environment variable management |
