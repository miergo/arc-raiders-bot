"""Centralized configuration loaded once from environment / .env file."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# This file lives at arc_raider_bot/config.py, so .parent.parent resolves to backend/.
BACKEND_DIR = Path(__file__).resolve().parent.parent

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
VERBOSE: bool = os.getenv("VERBOSE_TOOL_CALLS", "0") == "1"

CHROMA_DATA_DIR: str = os.getenv("CHROMA_DATA_DIR", str(BACKEND_DIR / "data" / "chroma"))
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
CACHE_SIMILARITY_THRESHOLD: float = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.75"))

MAX_SESSIONS: int = 200

MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "arc-raiders-rag")
