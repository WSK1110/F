"""
Centralized configuration for the RAG Chatbot.

Reads from environment variables with sensible defaults.
All hardcoded paths should be defined here, not scattered through code.
"""
import os
from pathlib import Path

# ── Base directories ────────────────────────────────────────────────────
# Override via env vars: CHROMA_DIR, DATA_DIR, HERMES_ENV
_BASE_DIR = Path(os.getenv("RAG_BASE_DIR", "/tmp/F_web"))

CHROMA_DIR = _BASE_DIR / "chroma_db"
CHROMA_COLLECTION = "10k_rag"

DATA_DIR = _BASE_DIR / "10k_files"
EVAL_DIR = _BASE_DIR / "chroma_eval"

# ── Hermes env file path ────────────────────────────────────────────────
HERMES_ENV_PATH = os.getenv("HERMES_ENV", os.path.expanduser("~/.hermes/.env"))

# ── Server ──────────────────────────────────────────────────────────────
HOST = os.getenv("RAG_HOST", "0.0.0.0")
PORT = int(os.getenv("RAG_PORT", "8000"))

# ── RAG config defaults ────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE = 850
DEFAULT_CHUNK_OVERLAP = 300
DEFAULT_SIMILARITY_TOP_K = 8
DEFAULT_RERANK_TOP_K = 12

# ── Hybrid retrieval ───────────────────────────────────────────────────
DEFAULT_USE_HYBRID = True
DEFAULT_SPARSE_WEIGHT = 0.50
DEFAULT_DENSE_WEIGHT = 0.80

# ── LLM defaults ───────────────────────────────────────────────────────
DEFAULT_LLM_PROVIDER = "deepseek"
DEFAULT_LLM_MODEL = "deepseek-chat"

DEFAULT_EMBEDDING_PROVIDER = "deepseek"
DEFAULT_EMBEDDING_MODEL = "deepseek-embedding"

# ── Cache ──────────────────────────────────────────────────────────────
QUERY_CACHE_TTL = 300  # seconds
MAX_CACHE_SIZE = 100

# ── Chat history ───────────────────────────────────────────────────────
MAX_CHAT_HISTORY = 50

# ── Performance ────────────────────────────────────────────────────────
PERFORMANCE_METRICS_MAX = 1000
