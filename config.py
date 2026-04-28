import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
CACHE_DIR = BASE_DIR / "cache"

for path in (DATA_DIR, INPUT_DIR, CACHE_DIR):
    path.mkdir(parents=True, exist_ok=True)

RAW_CHUNKS_FILE = CACHE_DIR / "raw_chunks.json"

MS_CLIENT_ID = os.environ.get("MS_CLIENT_ID", "")
MS_CLIENT_SECRET = os.environ.get("MS_CLIENT_SECRET", "")
MS_TENANT_ID = os.environ.get("MS_TENANT_ID", "")
MS_DRIVE_ID = os.environ.get("MS_DRIVE_ID", "")
MS_FOLDER_ID = os.environ.get("MS_FOLDER_ID", "")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
