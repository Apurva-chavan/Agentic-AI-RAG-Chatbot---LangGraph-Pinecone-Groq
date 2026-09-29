"""
tests_sample_queries.py
-----------------------
Runs 6 benchmark queries against the RAG pipeline and prints
structured JSON responses to verify grounding and context retrieval.
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from src.graph import query_rag

QUERIES = [
    "What is Agentic AI according to the eBook?",
    "How do AI agents differ from traditional automation systems?",
    "What are the core components of an Agentic Architecture?",
    "What role does memory play in Agentic AI workflows?",
    "What key challenges or limitations of Agentic AI are mentioned in the document?",
    # Out-of-scope groundedness test — system should refuse
    "Who won the 2022 FIFA World Cup?",
]


def run_tests():
    print("=" * 70)
    print("Agentic AI RAG Chatbot — Sample Query Validation")
    print("=" * 70)

    for i, query in enumerate(QUERIES, 1):
        print(f"\n[Query {i}] {query}")
        print("-" * 60)
        try:
            result = query_rag(query)
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"ERROR: {e}")
        print()


if __name__ == "__main__":
    run_tests()
