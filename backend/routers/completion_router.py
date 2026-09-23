"""
completion_router.py

FastAPI router for the Sentence Completion learning mode.

Endpoints:
    POST /completion/start     — generate a sentence-completion exercise
    POST /completion/evaluate  — score the user's answer semantically

Mounted in main.py with:
    app.include_router(completion_router)
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.completion_service import (
    start_completion_session,
    evaluate_completion,
)

logger = logging.getLogger(__name__)

completion_router = APIRouter(tags=["completion"])


# ============================================================
# REQUEST MODELS
# ============================================================

class CompletionStartRequest(BaseModel):
    difficulty: str = "medium"   # "easy" | "medium" | "hard"


class CompletionEvaluateRequest(BaseModel):
    session_id: str
    answer:     str


# ============================================================
# POST /completion/start
# ============================================================

@completion_router.post("/completion/start")
def completion_start(request: CompletionStartRequest):
    """
    Retrieve content from ChromaDB, generate a gapped sentence,
    store the expected answer server-side, and return:
        {
            "session_id":   "...",
            "sentence":     "The ______ is ...",
            "context_hint": "...",
            "difficulty":   "medium"
        }

    The expected answer is never sent to the client here.
    """
    difficulty = request.difficulty.lower().strip()
    if difficulty not in ("easy", "medium", "hard"):
        raise HTTPException(
            status_code=400,
            detail="difficulty must be 'easy', 'medium', or 'hard'.",
        )

    try:
        result = start_completion_session(difficulty)
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except RuntimeError as exc:
        err = str(exc)
        if "quota exhausted" in err.lower() or "RESOURCE_EXHAUSTED" in err:
            raise HTTPException(status_code=429, detail=err)
        raise HTTPException(status_code=400, detail=err)
    except Exception as exc:
        logger.exception("Unexpected error in completion_start")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start completion session: {exc}",
        )

    return result


# ============================================================
# POST /completion/evaluate
# ============================================================

@completion_router.post("/completion/evaluate")
def completion_evaluate(request: CompletionEvaluateRequest):
    """
    Semantically compare the user's answer to the server-stored expected
    concept and return structured feedback:
        {
            "score":            0-100,
            "status":           "correct|partially_correct|incorrect",
            "expected_concept": "...",
            "explanation":      "...",
            "feedback":         "..."
        }
    """
    if not request.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id cannot be empty.")

    if not request.answer.strip():
        raise HTTPException(status_code=400, detail="answer cannot be empty.")

    try:
        result = evaluate_completion(request.session_id, request.answer.strip())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except RuntimeError as exc:
        err = str(exc)
        if "quota exhausted" in err.lower() or "RESOURCE_EXHAUSTED" in err:
            raise HTTPException(status_code=429, detail=err)
        raise HTTPException(status_code=500, detail=err)
    except Exception as exc:
        logger.exception("Unexpected error in completion_evaluate")
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {exc}",
        )

    return result
