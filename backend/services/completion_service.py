"""
completion_service.py

Sentence Completion learning mode backend logic.

Public functions used by the completion router:
    start_completion_session(difficulty)  → create a session, return the gapped sentence
    evaluate_completion(session_id, answer) → semantically score the user's answer

Design:
    - Sessions stored in a plain in-memory dict (no DB, no Redis).
    - The expected answer concept is NEVER returned until after evaluation.
    - Evaluation is semantic: paraphrasing the correct concept is accepted.
    - Bounded retry (max 2 attempts) on Gemini 429, then raises RuntimeError.
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
# { session_id: { "sentence": str, "expected_concept": str,
#                 "context": str, "created_at": float } }
# ============================================================

_sessions: dict[str, dict] = {}

SESSION_TTL_SECONDS = 7200   # 2 hours


def _prune_old_sessions() -> None:
    """Discard sessions older than SESSION_TTL_SECONDS."""
    now = time.time()
    expired = [
        sid for sid, data in _sessions.items()
        if now - data["created_at"] > SESSION_TTL_SECONDS
    ]
    for sid in expired:
        del _sessions[sid]


# ============================================================
# DIFFICULTY GUIDES
# ============================================================

_DIFFICULTY_GUIDE = {
    "easy":   (
        "Choose a very common, clearly defined term or concept as the blank. "
        "The surrounding sentence should strongly hint at the answer."
    ),
    "medium": (
        "Choose a moderately important concept or term. "
        "The surrounding context should give useful but not obvious hints."
    ),
    "hard":   (
        "Choose a nuanced or precise concept. "
        "The surrounding context should be informative but require real understanding."
    ),
}


# ============================================================
# SENTENCE GENERATION PROMPT
# ============================================================

def _build_generation_prompt(context: str, difficulty: str) -> str:
    guide = _DIFFICULTY_GUIDE.get(difficulty, _DIFFICULTY_GUIDE["medium"])
    return f"""You are an educational exercise designer.

Using ONLY the document context below, create ONE sentence-completion exercise.

Rules:
1. Write one meaningful sentence that contains exactly one blank marked as "______".
2. The blank must replace exactly one important concept, term, or phrase from the context.
3. {guide}
4. Write a short context hint (1 sentence) that helps the student without giving away the answer.
5. Return ONLY valid JSON. No markdown fences. No extra text.

Required format:
{{
  "sentence": "The ______ is the process by which plants convert light into energy.",
  "expected_concept": "photosynthesis",
  "context_hint": "This concept is a fundamental process in plant biology."
}}

Rules for the JSON:
- "sentence" must contain exactly one "______" (6 underscores).
- "expected_concept" must be the exact word(s) that fill the blank.
- "context_hint" must NOT reveal the answer.
- Do not add markdown or extra keys.
- Return JSON only.

DOCUMENT CONTEXT:
{context}
"""


# ============================================================
# EVALUATION PROMPT
# ============================================================

def _build_evaluation_prompt(
    sentence: str,
    expected_concept: str,
    context: str,
    user_answer: str,
) -> str:
    return f"""You are an educational evaluator for a sentence-completion exercise.

The student was given a sentence with one blank and asked to fill it in.
Evaluate whether the student's answer conveys the correct concept.
Do NOT require exact wording — accept correct paraphrases and equivalent expressions.

Return ONLY valid JSON. No markdown fences. No extra text.

Required format:
{{
  "score": <integer 0-100>,
  "status": "<correct|partially_correct|incorrect>",
  "expected_concept": "<the ideal answer>",
  "explanation": "<why the answer is correct, partial, or incorrect>",
  "feedback": "<1-2 sentences of constructive, encouraging feedback to the student>"
}}

Scoring guide:
- 85-100  → correct: answer conveys the right concept (exact or paraphrase)
- 40-84   → partially_correct: answer is in the right area but incomplete or imprecise
- 0-39    → incorrect: answer is wrong, irrelevant, or empty

ORIGINAL SENTENCE (with blank):
{sentence}

EXPECTED CONCEPT:
{expected_concept}

STUDENT'S ANSWER:
{user_answer}

DOCUMENT CONTEXT (for grounding the evaluation):
{context}
"""


# ============================================================
# STRIP MARKDOWN FENCES
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
    Invoke the LLM singleton with up to 2 retries on 429.
    Returns the plain-text response. Raises RuntimeError on quota exhaustion.
    """
    for attempt in range(1, _LLM_MAX_RETRIES + 1):
        try:
            response = llm.invoke(prompt)
            return extract_text(response)
        except Exception as exc:
            err = str(exc)
            if "RESOURCE_EXHAUSTED" in err or "429" in err:
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
                raise


# ============================================================
# START COMPLETION SESSION
# ============================================================

def start_completion_session(difficulty: str) -> dict:
    """
    Retrieve content from ChromaDB, generate a sentence-completion exercise,
    store the expected concept server-side, and return the gapped sentence.

    Returns:
        {
            "session_id":    "...",
            "sentence":      "The ______ is ...",
            "context_hint":  "...",
            "difficulty":    "medium"
        }

    Raises:
        RuntimeError — no indexed content, or Gemini quota exhausted
        ValueError   — Gemini returned unparseable JSON
    """
    _prune_old_sessions()

    retriever = get_retriever()
    documents = retriever.invoke(
        "important concepts definitions terms key ideas facts"
    )

    if not documents:
        raise RuntimeError(
            "No indexed content found. "
            "Please upload and index a PDF before starting Sentence Completion."
        )

    context = format_context(documents)
    prompt  = _build_generation_prompt(context, difficulty)

    logger.info("Generating completion exercise (difficulty=%s).", difficulty)
    raw = _call_llm(prompt)
    raw = _strip_fences(raw)
    logger.debug("Raw generation response:\n%s", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"The AI returned invalid JSON for the exercise. "
            f"Raw (first 300 chars): {raw[:300]}"
        ) from exc

    sentence         = str(data.get("sentence", "")).strip()
    expected_concept = str(data.get("expected_concept", "")).strip()
    context_hint     = str(data.get("context_hint", "")).strip()

    if not sentence or "______" not in sentence:
        raise ValueError(
            "The AI did not produce a valid sentence with a blank. "
            "Please try again."
        )
    if not expected_concept:
        raise ValueError("The AI did not provide an expected concept. Please try again.")

    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "sentence":         sentence,
        "expected_concept": expected_concept,
        "context":          context,          # kept for evaluation grounding
        "created_at":       time.time(),
    }

    logger.info("Completion session created: %s", session_id)

    return {
        "session_id":   session_id,
        "sentence":     sentence,
        "context_hint": context_hint,
        "difficulty":   difficulty,
    }


# ============================================================
# EVALUATE COMPLETION
# ============================================================

def evaluate_completion(session_id: str, user_answer: str) -> dict:
    """
    Semantically compare the user's answer to the stored expected concept.

    Returns:
        {
            "score":            int (0-100),
            "status":           "correct" | "partially_correct" | "incorrect",
            "expected_concept": str,
            "explanation":      str,
            "feedback":         str
        }

    Raises:
        KeyError    — session not found / expired
        ValueError  — Gemini returned unparseable JSON
        RuntimeError — Gemini quota exhausted
    """
    session = _sessions.get(session_id)
    if session is None:
        raise KeyError(f"Session not found or expired: {session_id}")

    prompt = _build_evaluation_prompt(
        session["sentence"],
        session["expected_concept"],
        session["context"],
        user_answer,
    )

    logger.info("Evaluating completion for session %s.", session_id)
    raw = _call_llm(prompt)
    raw = _strip_fences(raw)
    logger.debug("Raw eval response:\n%s", raw)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"The AI returned invalid JSON during evaluation. "
            f"Raw (first 300 chars): {raw[:300]}"
        ) from exc

    # Normalise score
    try:
        score = int(result.get("score", 0))
        score = max(0, min(100, score))
    except (TypeError, ValueError):
        score = 0

    # Normalise status
    raw_status = str(result.get("status", "incorrect")).lower()
    if raw_status not in ("correct", "partially_correct", "incorrect"):
        raw_status = "incorrect"

    return {
        "score":            score,
        "status":           raw_status,
        "expected_concept": result.get("expected_concept", session["expected_concept"]),
        "explanation":      result.get("explanation", ""),
        "feedback":         result.get("feedback", ""),
    }
