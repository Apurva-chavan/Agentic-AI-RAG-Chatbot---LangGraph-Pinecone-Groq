"""
streamlit_app.py
----------------
Optional Streamlit web UI for the Agentic AI RAG Chatbot.
Run with: streamlit run streamlit_app.py
"""

import json
import sys
import os
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from src.graph import query_rag

st.set_page_config(page_title="Agentic AI RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Agentic AI RAG Chatbot")
st.caption("Answers grounded strictly in the *Agentic AI* eBook via LangGraph + Pinecone.")

SAMPLE_QUERIES = [
    "What is Agentic AI according to the eBook?",
    "How do AI agents differ from traditional automation systems?",
    "What are the core components of an Agentic Architecture?",
    "What role does memory play in Agentic AI workflows?",
    "What real-world industry use cases for Agentic AI are discussed in the eBook?",
    "Who won the 2022 FIFA World Cup?",
]

st.sidebar.header("📋 Sample Queries")
selected = st.sidebar.radio("Pick a sample:", ["(type your own)"] + SAMPLE_QUERIES)

user_query = st.text_input(
    "Your question:",
    value="" if selected == "(type your own)" else selected,
    placeholder="Ask something about Agentic AI...",
)

if st.button("🔍 Ask", type="primary", disabled=not bool(user_query.strip())):
    with st.spinner("Retrieving and generating answer..."):
        try:
            response = query_rag(user_query.strip())
        except Exception as e:
            st.error(f"Error: {e}")
            st.stop()

    score = response["confidence_score"]
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("💬 Answer")
        st.write(response["final_answer"])
    with col2:
        badge = "🟢 High" if score >= 0.80 else ("🟡 Medium" if score >= 0.50 else "🔴 Low / Out-of-scope")
        st.metric("Confidence Score", f"{score:.2f}", delta=badge)

    with st.expander("📄 Retrieved Context Chunks"):
        for i, chunk in enumerate(response["retrieved_context_chunks"], 1):
            st.markdown(f"**Chunk {i}:**")
            st.text(chunk)
            st.divider()

    st.subheader("📦 Full JSON Response Payload")
    st.code(json.dumps(response, indent=2), language="json")
