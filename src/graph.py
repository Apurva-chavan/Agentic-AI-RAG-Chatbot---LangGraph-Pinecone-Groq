"""
graph.py
--------
LangGraph RAG workflow:
  retrieve  → fetch top-k chunks from Pinecone (HuggingFace embeddings)
  generate  → produce a strictly grounded answer using Groq LLM (free)
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import List, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from src.config import (
    GROQ_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    TOP_K,
)


# ── State schema ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    question: str
    context: List[str]
    answer: str
    score: float


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_rag_graph(index_name: str = PINECONE_INDEX_NAME):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    llm = ChatGroq(model=LLM_MODEL, temperature=0, groq_api_key=GROQ_API_KEY)

    # ── Node 1: Retrieve ──────────────────────────────────────────────────────

    def retrieve_node(state: AgentState) -> dict:
        docs = retriever.invoke(state["question"])
        context_texts = [d.page_content for d in docs]
        return {"context": context_texts}

    # ── Node 2: Generate ──────────────────────────────────────────────────────

    def generate_node(state: AgentState) -> dict:
        context_str = "\n\n---\n\n".join(state["context"])

        prompt = (
            "You are a strict Q&A assistant. Answer the question using ONLY the context below.\n"
            "If the context does not contain enough information, respond with:\n"
            "'I cannot answer based on the provided document.'\n"
            "Do NOT use any external knowledge.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {state['question']}"
        )

        response = llm.invoke(prompt)
        confidence = 0.95 if len(state["context"]) > 0 else 0.0
        return {"answer": response.content, "score": confidence}

    # ── Assemble graph ────────────────────────────────────────────────────────

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


# ── Singleton ─────────────────────────────────────────────────────────────────

_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_rag_graph()
    return _graph


def query_rag(question: str) -> dict:
    initial_state: AgentState = {
        "question": question,
        "context": [],
        "answer": "",
        "score": 0.0,
    }
    result = get_graph().invoke(initial_state)
    return {
        "query": question,
        "final_answer": result["answer"],
        "retrieved_context_chunks": result["context"],
        "confidence_score": result["score"],
    }


if __name__ == "__main__":
    import json
    print(json.dumps(query_rag("What is Agentic AI according to the eBook?"), indent=2))
