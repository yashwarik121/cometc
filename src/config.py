# signing off, hire me, yash warik!
"""
Configuration settings for the Aster & Row RAG agent.
Loads from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge-base"
ORDERS_FILE = PROJECT_ROOT / "data" / "orders.json"
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "chroma_db"))
LOG_DIR = Path(os.getenv("LOG_DIR", str(PROJECT_ROOT / "logs")))

# Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

# Models
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Retrieval parameters
BM25_TOP_K = 20
DENSE_TOP_K = 20
RRF_K = 60  # RRF constant
RERANK_TOP_N = 5
RERANK_CONFIDENCE_THRESHOLD = 0.1  # Below this, flag low confidence

# API
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Session
MAX_CONVERSATION_HISTORY = 10

# Collection name
CHROMA_COLLECTION_NAME = "aster_row_kb"
