"""
app.py
------
FastAPI application exposing the Agentic AI RAG chatbot.

Endpoints:
  GET  /health  – liveness check
  POST /chat    – RAG query returning answer, context chunks, confidence score
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.graph import build_rag_graph
from src.config import PINECONE_INDEX_NAME

app = FastAPI(
    title="Agentic AI RAG API",
    description="LangGraph + Pinecone RAG pipeline grounded on the Agentic AI eBook.",
    version="1.0.0",
)

# Lazy-initialised so the graph is only built on first request
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_rag_graph(index_name=PINECONE_INDEX_NAME)
    return _graph


# ── Schemas ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    final_answer: str
    retrieved_context_chunks: list[str]
    confidence_score: float


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "Agentic AI RAG Chatbot API is running!",
        "docs": "/docs",
        "health": "/health",
        "chat": "POST /chat"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    try:
        initial_state = {
            "question": request.query.strip(),
            "context": [],
            "answer": "",
            "score": 0.0,
        }
        result = get_graph().invoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        query=request.query.strip(),
        final_answer=result["answer"],
        retrieved_context_chunks=result["context"],
        confidence_score=result["score"],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
