"""
ingestion.py
------------
Loads the Agentic AI eBook PDF, splits it into chunks,
generates HuggingFace embeddings (free, local),
and upserts into Pinecone via LangChain's PineconeVectorStore wrapper.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from src.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    PDF_PATH,
)


def _ensure_index_exists() -> None:
    """Create the Pinecone index if it does not already exist."""
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}' ...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("Index created.")
    else:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists.")


def run_ingestion(pdf_path: str = PDF_PATH, index_name: str = PINECONE_INDEX_NAME):
    """Full ingestion pipeline: load → chunk → embed → upsert."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"PDF not found at '{pdf_path}'.\n"
            "Place Ebook-Agentic-AI.pdf inside the data/ folder."
        )

    # 1. Load document pages
    print(f"Loading PDF from '{pdf_path}' ...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages.")

    # 2. Chunk document
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    # 3. Ensure Pinecone index exists
    _ensure_index_exists()

    # 4. Embed locally with HuggingFace and upsert
    print("Generating embeddings (local HuggingFace) and upserting to Pinecone ...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name,
    )
    print("Ingestion complete.")


if __name__ == "__main__":
    run_ingestion()
