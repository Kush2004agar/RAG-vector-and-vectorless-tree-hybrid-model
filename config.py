import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Base directories
BASE_DIR = Path("d:/experiment/rag+page index")
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

# Gemini API config
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "your_gemini_api_key")

# Retrieval Settings
RETRIEVER_TOP_K = 10 # Fixed: Increased from 1 to 10 for broader context retrieval
CHUNK_GROUP_SIZE = 5 # Number of chunks per Parent Node
