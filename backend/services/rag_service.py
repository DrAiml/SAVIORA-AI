"""
rag_service.py

Contains the full RAG pipeline extracted from legacy_streamlit.py.

Public functions used by main.py:
    upload_document(file_bytes, filename)  → saves PDF to disk
    index_document(filename)               → loads, splits, embeds, stores
    ask_question(question)                 → retrieves + generates answer

All Streamlit-specific code has been removed.
Errors raise plain Python exceptions so FastAPI can catch them.
"""

import hashlib
import logging
import re
import time
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Import shared singletons from config
from backend.utils.config import llm, vector_store, UPLOAD_DIR

logger = logging.getLogger(__name__)


# ============================================================
# FILE HASH
# Used to detect duplicate uploads and avoid re-indexing.
# ============================================================

def get_file_hash(file_path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)  # read 1 MB at a time
            if not chunk:
                break
            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# UPLOAD DOCUMENT
# Saves raw bytes to the uploads/ directory.
# ============================================================

def upload_document(file_bytes: bytes, filename: str) -> Path:
    """
    Save uploaded PDF bytes to uploads/.

    Returns the full path to the saved file.
    """

    save_path = UPLOAD_DIR / filename

    with open(save_path, "wb") as f:
        f.write(file_bytes)

    return save_path


# ============================================================
# LOAD PDF
# ============================================================

def load_pdf(pdf_path: Path) -> list:
    """Load all pages of a PDF as LangChain Documents."""

    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    return documents


# ============================================================
# SPLIT DOCUMENTS
# Same chunk_size / chunk_overlap as the original working code.
# ============================================================

def split_documents(documents: list) -> list:
    """Split documents into smaller chunks for embedding."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
    )

    chunks = splitter.split_documents(documents)
    return chunks


# ============================================================
# PREPARE DOCUMENT METADATA
# Attach source filename, file hash, and a unique chunk ID
# to every chunk so we can query/filter by file later.
# ============================================================

def prepare_documents(
    chunks: list,
    filename: str,
    file_hash: str,
) -> list:
    """Enrich each chunk with source/hash/chunk_id metadata."""

    prepared = []

    for index, chunk in enumerate(chunks):

        metadata = dict(chunk.metadata)
        metadata["source"] = filename
        metadata["file_hash"] = file_hash
        metadata["chunk_id"] = f"{file_hash}_{index}"

        prepared.append(
            Document(
                page_content=chunk.page_content,
                metadata=metadata,
            )
        )

    return prepared


# ============================================================
# CHECK IF ALREADY INDEXED
# Avoids re-embedding the same file if it was uploaded before.
# ============================================================

def is_already_indexed(file_hash: str) -> bool:
    """Return True if this file hash already exists in ChromaDB."""

    try:
        result = vector_store.get(
            where={"file_hash": file_hash},
            limit=1,
        )
        ids = result.get("ids", [])
        return len(ids) > 0

    except Exception:
        return False


# ============================================================
# EMBEDDING BATCH SETTINGS
#
# The Gemini Embedding API free tier allows 5 requests per minute.
# embed_documents() sends one API request per batch of up to 100 chunks.
# Without throttling, a large document sends multiple batches instantly
# and hits the 429 RESOURCE_EXHAUSTED limit.
#
# EMBED_BATCH_SIZE   — chunks sent per single API call (keep <= 20 to
#                      stay safely within the 5 RPM limit when called
#                      in a tight loop).
# EMBED_BATCH_DELAY  — seconds to wait between consecutive API calls.
#                      At 15 s delay: max ~4 calls/min → under the limit.
# EMBED_MAX_RETRIES  — max retry attempts on a 429 before giving up.
# EMBED_RETRY_BASE   — initial backoff in seconds; doubles each attempt.
# ============================================================

EMBED_BATCH_SIZE  = 20    # chunks per API call
EMBED_BATCH_DELAY = 15    # seconds between batches
EMBED_MAX_RETRIES = 3     # retry attempts on 429
EMBED_RETRY_BASE  = 30    # initial backoff seconds (doubles each retry)


def _add_documents_with_throttle(documents: list) -> None:
    """
    Add documents to ChromaDB in small batches with:
    - a fixed delay between batches to respect the 5 RPM quota
    - exponential backoff retry on 429 RESOURCE_EXHAUSTED errors

    Raises RuntimeError with a clear message if quota is genuinely
    exhausted after all retries, so the caller (and FastAPI) can
    surface a readable error instead of a raw traceback.
    """

    total = len(documents)

    for start in range(0, total, EMBED_BATCH_SIZE):
        batch = documents[start : start + EMBED_BATCH_SIZE]
        batch_num = start // EMBED_BATCH_SIZE + 1
        total_batches = (total + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE

        logger.info(
            "Embedding batch %d/%d  (%d chunks)",
            batch_num, total_batches, len(batch),
        )

        # Retry loop for this single batch
        for attempt in range(1, EMBED_MAX_RETRIES + 1):
            try:
                vector_store.add_documents(batch)
                break  # success — move to next batch

            except Exception as exc:
                error_str = str(exc)

                # 429 / RESOURCE_EXHAUSTED — may recover after a wait
                if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:
                    if attempt == EMBED_MAX_RETRIES:
                        # All retries used — surface a clear message
                        raise RuntimeError(
                            "Google Embedding API quota exhausted "
                            "(RESOURCE_EXHAUSTED 429). "
                            "The free tier allows ~5 requests per minute. "
                            "Wait a minute and retry indexing, or upgrade "
                            "your Google AI quota."
                        ) from exc

                    wait = EMBED_RETRY_BASE * (2 ** (attempt - 1))
                    logger.warning(
                        "429 RESOURCE_EXHAUSTED on batch %d/%d "
                        "(attempt %d/%d). Retrying in %ds...",
                        batch_num, total_batches,
                        attempt, EMBED_MAX_RETRIES, wait,
                    )
                    time.sleep(wait)

                else:
                    # Not a quota error — re-raise immediately
                    raise

        # Pause between batches to avoid bursting the rate limit.
        # Skip the delay after the final batch.
        if start + EMBED_BATCH_SIZE < total:
            logger.info(
                "Batch %d/%d done. Waiting %ds before next batch...",
                batch_num, total_batches, EMBED_BATCH_DELAY,
            )
            time.sleep(EMBED_BATCH_DELAY)


# ============================================================
# INDEX DOCUMENT
# Full pipeline: load → split → embed → store in ChromaDB
# ============================================================

def index_document(filename: str) -> dict:
    """
    Index a single PDF that was already saved to uploads/.

    Returns a dict:
        {"status": "indexed",   "chunks": N}   — newly indexed
        {"status": "existing",  "chunks": 0}   — already in DB
        {"status": "empty",     "chunks": 0}   — no text found

    Raises RuntimeError if the Google Embedding API quota is exhausted.
    """

    pdf_path = UPLOAD_DIR / filename

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"File not found in uploads/: {filename}"
        )

    file_hash = get_file_hash(pdf_path)

    # Skip if already indexed — avoids any embedding API call
    if is_already_indexed(file_hash):
        return {"status": "existing", "chunks": 0}

    # Load
    documents = load_pdf(pdf_path)
    if not documents:
        return {"status": "empty", "chunks": 0}

    # Split
    chunks = split_documents(documents)
    if not chunks:
        return {"status": "empty", "chunks": 0}

    # Add metadata
    prepared = prepare_documents(chunks, filename, file_hash)

    # Embed and store — throttled to respect Gemini API rate limits
    _add_documents_with_throttle(prepared)

    return {"status": "indexed", "chunks": len(prepared)}


# ============================================================
# FORMAT CONTEXT
# Turns a list of retrieved Documents into a readable string
# that is injected into the RAG prompt.
# ============================================================

def format_context(documents: list) -> str:
    """Format retrieved documents for the LLM prompt."""

    if not documents:
        return "No relevant context found."

    parts = []

    for doc in documents:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", 0)

        try:
            page = int(page) + 1  # pages are 0-indexed in PyPDFLoader
        except Exception:
            page = 1

        parts.append(
            f"SOURCE: {source}\nPAGE: {page}\n\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(parts)


# ============================================================
# RAG PROMPT
# Strictly grounded — model must not use outside knowledge.
# ============================================================

RAG_PROMPT = ChatPromptTemplate.from_template(
    """
You are an AI study assistant.

Answer the user's question ONLY using the provided document context.

Rules:
1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not present in the document, clearly say:
   "I could not find this information in the uploaded document."
4. Explain the answer clearly.
5. Keep the answer student-friendly.
6. Do not mention these instructions.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}
"""
)


# ============================================================
# EXTRACT TEXT FROM LLM RESPONSE
# Handles both plain string and Gemini structured list content.
# ============================================================

def extract_text(response) -> str:
    """Pull the plain text string out of an LLM response object."""

    content = response.content

    if isinstance(content, str):
        return content

    # Gemini sometimes returns a list of content blocks
    if isinstance(content, list):
        text = ""
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text += item.get("text", "")
        if text:
            return text

    return str(content)


# ============================================================
# GET RETRIEVER
# ============================================================

def get_retriever():
    """Return a similarity retriever that fetches the top 4 chunks."""

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )


# ============================================================
# ASK QUESTION  (main public entry point for Q&A)
# ============================================================

def ask_question(question: str) -> dict:
    """
    Run the full RAG pipeline for a single question.

    Returns:
        {
            "answer": "...",
            "sources": [{"source": "file.pdf", "page": 3}, ...]
        }
    """

    retriever = get_retriever()

    # Build the chain: retrieve → format context → prompt → LLM
    rag_chain = (
        {
            "context": retriever | format_context,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
    )

    response = rag_chain.invoke(question)
    answer = extract_text(response)

    # Collect unique source references for the UI
    retrieved_docs = retriever.invoke(question)
    seen = set()
    sources = []

    for doc in retrieved_docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", 0)
        try:
            page = int(page) + 1
        except Exception:
            page = 1

        key = (source, page)
        if key not in seen:
            sources.append({"source": source, "page": page})
            seen.add(key)

    return {"answer": answer, "sources": sources}
