
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Gemini configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# Embedding configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking configuration
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400

# Retrieval configuration
TOP_K = 5

# Chroma collection
COLLECTION_NAME = "ai_ml_knowledge"
