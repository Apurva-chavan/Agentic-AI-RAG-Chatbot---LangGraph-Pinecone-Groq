"""
config.py
---------
Centralised environment setup and project-wide constants.
"""

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY: str = os.environ["GROQ_API_KEY"]
PINECONE_API_KEY: str = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")

EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384
LLM_MODEL: str = "openai/gpt-oss-20b"

CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 200
TOP_K: int = 3

PDF_PATH: str = os.getenv("PDF_PATH", "data/Ebook-Agentic-AI.pdf")
