import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
VECTOR_DIR = BASE_DIR / "vector_store"
REPORTS_DIR = BASE_DIR / "reports"

# Data files
RAW_CSV = os.environ.get("RAW_CSV", str(RAW_DIR / "complaints_sample.csv"))
FILTERED_CSV = PROCESSED_DIR / "filtered_complaints.csv"
CHUNKS_CSV = PROCESSED_DIR / "chunks.csv"

# Vector store files
FAISS_INDEX_PATH = VECTOR_DIR / "index.faiss"
META_PATH = VECTOR_DIR / "meta.pkl"

# Products of interest
ALLOWED_PRODUCTS = {
    "Credit card",
    "Personal loan",
    "Buy Now, Pay Later",
    "Savings account",
    "Money transfer",
    "Money transfers",  # handle plural variant
}

# Embedding model
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Retrieval
TOP_K = int(os.environ.get("TOP_K", "5"))

# LLM provider selection
# Options: "openai" (needs OPENAI_API_KEY) or "transformers"
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "transformers")
TRANSFORMERS_MODEL = os.environ.get("TRANSFORMERS_MODEL", "google/flan-t5-base")
MAX_INPUT_TOKENS = int(os.environ.get("MAX_INPUT_TOKENS", "2048"))

# Streamlit
APP_TITLE = "CrediTrust Complaint Intelligence (RAG)"
