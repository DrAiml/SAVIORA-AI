"""
recall_router.py

FastAPI router for the Passage Recall learning mode.

Endpoints:
    POST /recall/start     — create a session and return a passage to the client
    POST /recall/evaluate  — score the user's recalled text against the stored passage

Mounted in main.py with:
    app.include_router(recall_router)
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.recall_service import start_recall_session, evaluate_recall

logger = logging.getLogger(__name__)

recall_router = APIRouter(tags=["recall"])


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class RecallStartRequest(BaseModel):
    difficulty: str = "medium"   # "easy" | "medium" | "hard"


class RecallEvaluateRequest(BaseModel):
    session_id: str
    response:   str


# ============================================================
# POST /recall/start
# ============================================================

@recall_router.post("/recall/start")
def recall_start(request: RecallStartRequest):
    """
    Retrieve content from ChromaDB, generate an educational passage,
    store it server-side, and return:
        { "session_id": "...", "passage": "...", "duration_seconds": 30 }

    The reference passage is stored server-side only.
    The client receives it once here so the user can read it;
    it is NOT sent again during evaluation.
    """

    difficulty = request.difficulty.lower().strip()
    if difficulty not in ("easy", "medium", "hard"):
        raise HTTPException(
            status_code=400,
            detail="difficulty must be 'easy', 'medium', or 'hard'.",
        )

    try:
        result = start_recall_session(difficulty)
    except RuntimeError as exc:
        error_str = str(exc)
        if "quota exhausted" in error_str.lower() or "RESOURCE_EXHAUSTED" in error_str:
            raise HTTPException(status_code=429, detail=error_str)
        raise HTTPException(status_code=400, detail=error_str)
    except Exception as exc:
        logger.exception("Unexpected error in recall_start")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start recall session: {exc}",
        )

    return result


# ============================================================
# POST /recall/evaluate
# ============================================================

@recall_router.post("/recall/evaluate")
def recall_evaluate(request: RecallEvaluateRequest):
    """
    Compare the user's recalled text against the server-stored passage
    using Gemini and return structured feedback:
        {
            "score":             0–100,
            "summary":           "...",
            "remembered_points": [...],
            "missed_points":     [...],
            "feedback":          "..."
        }

    The reference passage is retrieved from the server session — it is
    never sent by the client, ensuring the client cannot trivially cheat
    by re-submitting the passage verbatim.
    """

    if not request.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id cannot be empty.")

    if not request.response.strip():
        raise HTTPException(status_code=400, detail="response cannot be empty.")

    try:
        result = evaluate_recall(request.session_id, request.response.strip())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        # LLM returned bad JSON
        raise HTTPException(status_code=502, detail=str(exc))
    except RuntimeError as exc:
        error_str = str(exc)
        if "quota exhausted" in error_str.lower() or "RESOURCE_EXHAUSTED" in error_str:
            raise HTTPException(status_code=429, detail=error_str)
        raise HTTPException(status_code=500, detail=error_str)
    except Exception as exc:
        logger.exception("Unexpected error in recall_evaluate")
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {exc}",
        )

    return result
