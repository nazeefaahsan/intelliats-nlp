"""Application settings."""
from pathlib import Path
import os
BASE_DIR = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
SECRET_KEY = os.environ.get("SECRET_KEY") or os.urandom(32)
DATABASE_PATH = BASE_DIR / "instance" / "ats.db"
SEMANTIC_ENABLED = os.environ.get("INTELLIATS_SEMANTIC", "0").strip().lower() in {"1", "true", "yes", "on"}
