"""
config.py

Central place for:
- Loading the .env file
- Creating the LLM instance
- Creating the Embeddings instance
- Creating the ChromaDB vector store instance

These are created ONCE as module-level singletons so that
every service can import and reuse them without re-initializing.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_chroma import Chroma


# ============================================================
# Load .env
# The .env file lives at the project root (AI_Study_Tutor/.env)
# ============================================================

# backend/utils/config.py  →  go up two levels to reach project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

load_dotenv(PROJECT_ROOT / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise EnvironmentError(
        "GOOGLE_API_KEY not found.\n"
        "Create a .env file at the project root with:\n"
        "GOOGLE_API_KEY=your_api_key_here"
    )


# ============================================================
# Paths
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BACKEND_DIR / "uploads"
CHROMA_DIR = BACKEND_DIR / "chroma_db"

UPLOAD_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)


# ============================================================
# Singletons — created once, reused everywhere
# ============================================================

# LLM — temperature=0 keeps answers deterministic/factual
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0,
)

# Embeddings
# The installed langchain-google-genai 4.4.0 package shows the model
# name used in all examples is "gemini-embedding-2-preview".
# The original final_v3.py used "gemini-embedding-2" (without -preview).
# These are passed directly to the Google GenAI API — the package does
# NOT validate or rewrite model names. Whether "gemini-embedding-2"
# resolves on the API side depends on what Google has published under
# that exact name. The installed package accepts any string you provide.
#
# CONFIGURED AS REQUESTED: "gemini-embedding-2"
# If the API returns a 404 / model-not-found error at runtime, the
# correct name to use is "gemini-embedding-2-preview" (as shown in
# the langchain-google-genai 4.4.0 source docstring examples).
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
)

# ChromaDB vector store
vector_store = Chroma(
    collection_name="study_tutor",
    persist_directory=str(CHROMA_DIR),
    embedding_function=embeddings,
)
