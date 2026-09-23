"""
main.py

FastAPI backend for AI Study Tutor.

Endpoints:
    GET  /health        — liveness check
    POST /upload        — save an uploaded PDF to disk
    POST /index         — embed and store a PDF in ChromaDB
    POST /ask           — answer a question using RAG

Run with:
    cd AI_Study_Tutor
    E:\\Langchain_Projects\\venv\\Scripts\\uvicorn.exe backend.main:app --reload --port 8000
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# All RAG logic lives in the service — main.py just wires HTTP to it.
from backend.services.rag_service import (
    upload_document,
    index_document,
    ask_question,
)
from backend.services.quiz_service import generate_quiz
from backend.routers.recall_router import recall_router
from backend.routers.completion_router import completion_router


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="AI Study Tutor",
    description="RAG-powered study assistant backed by Gemini + ChromaDB",
    version="1.0.0",
)


# ============================================================
# CORS
# Allow any localhost origin so the React frontend (any port)
# can call this API without browser CORS errors.
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # React default (Create React App)
        "http://localhost:5173",   # Vite default
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://localhost:5500",   # VS Code Live Server default
        "http://127.0.0.1:5500",
        "null", 
        "https://saviora-ai-1.onrender.com",                   # file:// origin — plain HTML opened from disk
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(recall_router)
app.include_router(completion_router)


# ============================================================
# REQUEST MODELS
# ============================================================

class IndexRequest(BaseModel):
    filename: str


class AskRequest(BaseModel):
    question: str


class QuizRequest(BaseModel):
    num_questions: int = 5   # default to 5 if not specified


# ============================================================
# GET /health
# ============================================================

@app.get("/health")
def health():
    """Quick liveness check — confirms the server is running."""
    return {"status": "ok"}


# ============================================================
# POST /upload
# ============================================================

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    Accept a PDF file and save it to the uploads/ directory.

    The filename is preserved exactly as uploaded.
    Returns the filename so the client can pass it to /index.
    """

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    file_bytes = await file.read()

    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    try:
        saved_path = upload_document(file_bytes, file.filename)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file: {e}",
        )

    return {
        "message": "File uploaded successfully.",
        "filename": file.filename,
        "size_bytes": len(file_bytes),
    }


# ============================================================
# POST /index
# ============================================================

@app.post("/index")
def index(request: IndexRequest):
    """
    Load, split, embed, and store a previously uploaded PDF.

    Pass the exact filename returned by /upload.
    """

    try:
        result = index_document(request.filename)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        # Quota-exhausted errors surface as RuntimeError with a clear message
        error_str = str(e)
        if "RESOURCE_EXHAUSTED" in error_str or "quota exhausted" in error_str.lower():
            raise HTTPException(status_code=429, detail=str(e))
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Indexing failed: {e}",
        )

    # Build a human-readable message alongside the raw status
    messages = {
        "indexed":  f"Indexed successfully — {result['chunks']} chunks stored.",
        "existing": "This document is already indexed. No changes made.",
        "empty":    "No readable text found in this PDF.",
    }

    return {
        "filename": request.filename,
        "status":   result["status"],
        "chunks":   result["chunks"],
        "message":  messages.get(result["status"], result["status"]),
    }


# ============================================================
# POST /ask
# ============================================================

@app.post("/ask")
def ask(request: AskRequest):
    """
    Answer a question using the RAG pipeline.

    Retrieves relevant chunks from ChromaDB, injects them into
    the Gemini prompt, and returns the grounded answer.
    """

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        result = ask_question(request.question)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate answer: {e}",
        )

    return {
        "question": request.question,
        "answer":   result["answer"],
        "sources":  result["sources"],
    }


# ============================================================
# POST /quiz
# ============================================================

@app.post("/quiz")
def quiz(request: QuizRequest):
    """
    Generate MCQ questions from the indexed document.

    The questions are grounded entirely in ChromaDB — the LLM
    only sees context retrieved from the user's uploaded PDF.

    Request:  { "num_questions": 5 }
    Response: { "questions": [ { "question", "options", "answer", "explanation" } ] }
    """

    if request.num_questions < 1 or request.num_questions > 20:
        raise HTTPException(
            status_code=400,
            detail="num_questions must be between 1 and 20.",
        )

    try:
        questions = generate_quiz(request.num_questions)
    except RuntimeError as e:
        # No document indexed yet, or ChromaDB is empty
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        # LLM returned bad JSON or failed validation
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Quiz generation failed: {e}",
        )

    return {"questions": questions}
