"""
recall_service.py

Passage Recall learning mode backend logic.

Public functions used by the recall router:
    start_recall_session(difficulty)  → creates a session, returns passage + session_id
    evaluate_recall(session_id, response)  → compares recall against the stored passage

Design decisions:
    - Sessions are stored in a plain in-memory dict (no DB, no Redis).
      Sessions expire when the process restarts — that is acceptable for
      this single-server, no-auth setup.
    - The reference passage is NEVER returned to the client after /recall/start.
    - Evaluation uses the existing Gemini LLM singleton.
    - Bounded retry (max 2 attempts) on 429 from Gemini, then raises.
    - Reuses get_retriever / format_context / extract_text from rag_service.
"""

import json
import logging
import re
import time
import uuid

from backend.services.rag_service import (
    get_retriever,
    format_context,
    extract_text,
)
from backend.utils.config import llm

logger = logging.getLogger(__name__)


# ============================================================
# IN-MEMORY SESSION STORE
# { session_id: {"passage": str, "created_at": float} }
# ============================================================

_sessions: dict[str, dict] = {}

# Sessions older than 2 hours are silently discarded
SESSION_TTL_SECONDS = 7200


def _prune_old_sessions() -> None:
    """Remove sessions that have exceeded SESSION_TTL_SECONDS."""
    now = time.time()
    expired = [
        sid for sid, data in _sessions.items()
        if now - data["created_at"] > SESSION_TTL_SECONDS
    ]
    for sid in expired:
        del _sessions[sid]


# ============================================================
# PASSAGE GENERATION PROMPT
# ============================================================

def _build_passage_prompt(context: str, difficulty: str) -> str:
    """Return the prompt that asks Gemini to write a study passage."""

    difficulty_guide = {
        "easy":   "Use simple language. Focus on one or two core ideas. 80–120 words.",
        "medium": "Cover 3–4 key concepts clearly. 120–180 words.",
        "hard":   "Cover multiple interrelated concepts with some detail. 180–250 words.",
    }.get(difficulty, "Cover 3–4 key concepts clearly. 120–180 words.")

    return f"""You are an educational content writer.

Using ONLY the document context provided below, write a single coherent educational
passage suitable for a memory recall exercise.

Guidelines:
- {difficulty_guide}
- Write in clear, flowing prose (not bullet points).
- Include only factual information from the context.
- Do not invent any information not present in the context.
- Do not include a title or heading.
- Output ONLY the passage text — no extra commentary, no markdown.

DOCUMENT CONTEXT:
{context}
"""


# ============================================================
# EVALUATION PROMPT
# ============================================================

def _build_eval_prompt(passage: str, user_response: str) -> str:
    """Return the prompt that asks Gemini to evaluate recall quality."""

    return f"""You are an educational evaluator assessing a student's memory recall exercise.

The student was shown an educational passage, then asked to write down everything they remembered.
Evaluate their recall based on CONCEPTUAL ACCURACY and COMPLETENESS — not on exact wording.
Do not penalise the student for paraphrasing or using different words to express the same idea.

Return ONLY valid JSON. No markdown fences. No extra text.

Required format:
{{
  "score": <integer 0-100>,
  "summary": "<1-2 sentence overall assessment>",
  "remembered_points": ["<point 1>", "<point 2>", ...],
  "missed_points": ["<point 1>", "<point 2>", ...],
  "feedback": "<2-3 sentences of constructive, encouraging feedback>"
}}

Scoring guide:
- 90-100: Recalled nearly all key concepts accurately
- 70-89:  Recalled most concepts with minor gaps
- 50-69:  Recalled some concepts but missed several important ones
- 30-49:  Recalled a few concepts; significant gaps remain
- 0-29:   Very little accurate recall

ORIGINAL PASSAGE:
{passage}

STUDENT'S RECALL:
{user_response}
"""


# ============================================================
# STRIP MARKDOWN FENCES
# Gemini sometimes wraps JSON in ```json ... ``` despite instructions.
# ============================================================

def _strip_fences(raw: str) -> str:
    raw = re.sub(r"^```json\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"^```\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


# ============================================================
# CALL LLM WITH BOUNDED RETRY
# Max 2 attempts on 429; all other errors raise immediately.
# ============================================================

_LLM_MAX_RETRIES = 2
_LLM_RETRY_BASE_SECONDS = 15


def _call_llm(prompt: str) -> str:
    """
    Invoke the LLM with up to _LLM_MAX_RETRIES retries on 429.

    Returns the plain-text response string.
    Raises RuntimeError on quota exhaustion after all retries.
    Raises other exceptions immediately.
    """
    for attempt in range(1, _LLM_MAX_RETRIES + 1):
        try:
            response = llm.invoke(prompt)
            return extract_text(response)
        except Exception as exc:
            error_str = str(exc)
            if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:
                if attempt == _LLM_MAX_RETRIES:
                    raise RuntimeError(
                        "Gemini API quota exhausted (429 RESOURCE_EXHAUSTED). "
                        "Please wait a minute and try again."
                    ) from exc
                wait = _LLM_RETRY_BASE_SECONDS * attempt
                logger.warning(
                    "Gemini 429 on attempt %d/%d — retrying in %ds",
                    attempt, _LLM_MAX_RETRIES, wait,
                )
                time.sleep(wait)
            else:
                raise  # Not a quota error — propagate immediately


# ============================================================
# START RECALL SESSION
# ============================================================

def start_recall_session(difficulty: str) -> dict:
    """
    Retrieve content from ChromaDB, generate an educational passage,
    store it server-side with a new session_id, and return the session.

    Returns:
        {
            "session_id": "...",
            "passage":    "...",
            "duration_seconds": 30
        }

    Raises:
        RuntimeError  — if ChromaDB has no indexed content
        RuntimeError  — if Gemini quota is exhausted
    """

    _prune_old_sessions()

    retriever = get_retriever()

    # Broad retrieval query to pull varied content
    documents = retriever.invoke(
        "important concepts definitions facts examples key points"
    )

    if not documents:
        raise RuntimeError(
            "No indexed content found. "
            "Please upload and index a PDF before starting Passage Recall."
        )

    context = format_context(documents)
    prompt  = _build_passage_prompt(context, difficulty)

    logger.info("Generating recall passage (difficulty=%s).", difficulty)
    raw_passage = _call_llm(prompt)

    passage = raw_passage.strip()
    if not passage:
        raise RuntimeError(
            "The AI returned an empty passage. Please try again."
        )

    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "passage":    passage,
        "created_at": time.time(),
    }

    logger.info("Recall session created: %s", session_id)

    return {
        "session_id":       session_id,
        "passage":          passage,
        "duration_seconds": 30,
    }


# ============================================================
# EVALUATE RECALL
# ============================================================

def evaluate_recall(session_id: str, user_response: str) -> dict:
    """
    Compare the user's recalled text against the stored passage using Gemini.

    Returns:
        {
            "score":              int (0–100),
            "summary":            str,
            "remembered_points":  list[str],
            "missed_points":      list[str],
            "feedback":           str
        }

    Raises:
        KeyError    — if session_id is not found or has expired
        ValueError  — if Gemini returns unparseable JSON
        RuntimeError — if Gemini quota is exhausted
    """

    session = _sessions.get(session_id)
    if session is None:
        raise KeyError(f"Session not found: {session_id}")

    passage = session["passage"]

    prompt  = _build_eval_prompt(passage, user_response)
    logger.info("Evaluating recall for session %s.", session_id)

    raw  = _call_llm(prompt)
    raw  = _strip_fences(raw)

    logger.debug("Raw eval response:\n%s", raw)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("JSON parse failed for eval. Raw:\n%s", raw)
        raise ValueError(
            f"The AI returned invalid JSON during evaluation. "
            f"Raw response (first 300 chars): {raw[:300]}"
        ) from exc

    # Normalise / fill missing keys with safe defaults
    score = result.get("score", 0)
    try:
        score = int(score)
        score = max(0, min(100, score))
    except (TypeError, ValueError):
        score = 0

    return {
        "score":             score,
        "summary":           result.get("summary",           "Evaluation complete."),
        "remembered_points": result.get("remembered_points", []),
        "missed_points":     result.get("missed_points",     []),
        "feedback":          result.get("feedback",          ""),
    }
