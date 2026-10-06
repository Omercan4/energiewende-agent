"""All settings in one place. Values come from environment variables or the .env file."""

import os

from dotenv import load_dotenv

load_dotenv()  # reads .env in the current folder, if it exists

# Folder where downloaded API responses are cached.
CACHE_DIR = os.environ.get("CACHE_DIR", "data/cache")

# Key for the Bundestag DIP API.
DIP_API_KEY = os.environ.get("DIP_API_KEY", "")

# Folder where the search index (Chroma database) is stored.
INDEX_DIR = os.environ.get("INDEX_DIR", "data/chroma")

# Local HuggingFace model that turns text into vectors.
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

# The LLM. Any OpenAI-compatible endpoint works.
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.5-flash")

# The model that grades text answers in the evaluation (another model family than the agent).
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-haiku-4.5")

# Where MLflow saves evaluation runs.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

# Optional password for POST /ask (header X-API-Key). Empty means no password, e.g. locally.
APP_API_KEY = os.environ.get("APP_API_KEY", "")
