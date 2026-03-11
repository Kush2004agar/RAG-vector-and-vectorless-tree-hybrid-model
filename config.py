import os
from pathlib import Path

from dotenv import load_dotenv

# Repo root (directory containing this file)
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env in repo root (not committed)
load_dotenv(BASE_DIR / ".env")
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
CACHE_DIR = BASE_DIR / "cache"
VECTOR_DB_DIR = BASE_DIR / "vector_db"

# Create directories if they don't exist
for d in [INPUT_DIR, CACHE_DIR, VECTOR_DB_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# File paths
PROCESSED_FILES_TRACKER = CACHE_DIR / "processed_files.json"
KNOWLEDGE_GRAPH_FILE = CACHE_DIR / "knowledge_graph.ttl"
CHAT_HISTORY_FILE = CACHE_DIR / "chat_history.json"
TREES_DIR = CACHE_DIR / "chunk_trees"
TREES_DIR.mkdir(parents=True, exist_ok=True)

# Microsoft Graph API config (Environment Variables should be set)
MS_CLIENT_ID = os.environ.get("MS_CLIENT_ID", "your_client_id")
MS_CLIENT_SECRET = os.environ.get("MS_CLIENT_SECRET", "your_client_secret")
MS_TENANT_ID = os.environ.get("MS_TENANT_ID", "your_tenant_id")
MS_DRIVE_ID = os.environ.get("MS_DRIVE_ID", "your_drive_id")
MS_FOLDER_ID = os.environ.get("MS_FOLDER_ID", "your_folder_id")

# Gemini API config (strip whitespace and surrounding quotes so key is valid)
_raw_key = os.environ.get("GEMINI_API_KEY", "your_gemini_api_key")
GEMINI_API_KEY = _raw_key.strip().strip('"').strip("'") if _raw_key else ""


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# Retrieval Settings
# For 1000+ PDFs: increase ROOT_RETRIEVER_TOP_K (e.g. 15–20), INITIAL_CONTEXT_CHUNK_COUNT (e.g. 50–80),
# FALLBACK_CONTEXT_CHUNK_COUNT (e.g. 80–100), MAX_PARENT_CANDIDATES (e.g. 15–20).
RETRIEVER_TOP_K = int(os.environ.get("RETRIEVER_TOP_K", "10"))  # Legacy
ROOT_RETRIEVER_TOP_K = int(os.environ.get("ROOT_RETRIEVER_TOP_K", "12"))  # PDFs to consider (raise for 1000+ docs)
PARENT_RETRIEVER_TOP_K = int(os.environ.get("PARENT_RETRIEVER_TOP_K", "8"))
DIRECT_CHUNK_RETRIEVER_TOP_K = int(os.environ.get("DIRECT_CHUNK_RETRIEVER_TOP_K", "10"))
# Chunks sent to Gemini: first attempt, then fallback if "not found"
INITIAL_CONTEXT_CHUNK_COUNT = int(os.environ.get("INITIAL_CONTEXT_CHUNK_COUNT", "50"))
FALLBACK_CONTEXT_CHUNK_COUNT = int(os.environ.get("FALLBACK_CONTEXT_CHUNK_COUNT", "80"))
FINAL_CONTEXT_CHUNK_COUNT = INITIAL_CONTEXT_CHUNK_COUNT
MAX_PARENT_CANDIDATES = int(os.environ.get("MAX_PARENT_CANDIDATES", "15"))
CHUNK_GROUP_SIZE = int(os.environ.get("CHUNK_GROUP_SIZE", "5"))  # Number of chunks per Parent Node

# Ingestion/chunking controls
CHUNK_MIN_CHARS = int(os.environ.get("CHUNK_MIN_CHARS", "160"))
CHUNK_TARGET_CHARS = int(os.environ.get("CHUNK_TARGET_CHARS", "900"))
CHUNK_OVERLAP_CHARS = int(os.environ.get("CHUNK_OVERLAP_CHARS", "120"))

# Query router / reranker controls
ENABLE_QUERY_ROUTER = _env_bool("ENABLE_QUERY_ROUTER", True)
ENABLE_RERANKING = _env_bool("ENABLE_RERANKING", True)
ENABLE_MULTI_QUERY_RETRIEVAL = _env_bool("ENABLE_MULTI_QUERY_RETRIEVAL", True)
MULTI_QUERY_COUNT = int(os.environ.get("MULTI_QUERY_COUNT", "3"))
RERANKER_MODEL_NAME = os.environ.get(
    "RERANKER_MODEL_NAME",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
).strip()
RERANK_CANDIDATE_POOL_SIZE = int(os.environ.get("RERANK_CANDIDATE_POOL_SIZE", "30"))
RERANKED_TOP_K = int(os.environ.get("RERANKED_TOP_K", "5"))
TREE_SCORE_WEIGHT = float(os.environ.get("TREE_SCORE_WEIGHT", "1.0"))
SEMANTIC_SCORE_WEIGHT = float(os.environ.get("SEMANTIC_SCORE_WEIGHT", "1.0"))
RERANK_SCORE_WEIGHT = float(os.environ.get("RERANK_SCORE_WEIGHT", "1.5"))
CONTEXT_COMPRESSION_MAX_CHARS = int(os.environ.get("CONTEXT_COMPRESSION_MAX_CHARS", "900"))
CONTEXT_MIN_RELEVANCE_SCORE = float(os.environ.get("CONTEXT_MIN_RELEVANCE_SCORE", "2.0"))

# Retrieval result cache
ENABLE_RETRIEVAL_CACHE = _env_bool("ENABLE_RETRIEVAL_CACHE", True)
RETRIEVAL_CACHE_SIZE = int(os.environ.get("RETRIEVAL_CACHE_SIZE", "256"))

# Excel QA pipeline: max questions to process (None = all rows in the Excel)
_max_q = (os.environ.get("MAX_QUESTIONS") or "all").strip().lower()
MAX_QUESTIONS = None if _max_q in ("none", "", "0", "all") else int(_max_q)
SAVE_PROGRESS_EVERY_N = int(os.environ.get("SAVE_PROGRESS_EVERY_N", "50"))  # Save partial Excel every N questions

# Optional: set a default questions Excel (leave None to auto-pick).
# Prefer passing `-i/--input` to `run_qa_pipeline.py` for explicit targeting.
DEFAULT_QUESTIONS_EXCEL: str | None = None
