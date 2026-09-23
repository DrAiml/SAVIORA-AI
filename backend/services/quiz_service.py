"""
quiz_service.py

Generates multiple-choice quizzes from the indexed document.

The quiz is grounded entirely in the ChromaDB vector store —
the LLM only sees context retrieved from the user's uploaded PDF.

Public function used by main.py:
    generate_quiz(num_questions)  →  list of question dicts
"""

import json
import logging
import re

# Reuse the shared singletons and helpers already established in rag_service.
# Do NOT create a second vector store or LLM instance.
from backend.services.rag_service import (
    get_retriever,
    format_context,
    extract_text,
)
from backend.utils.config import llm

logger = logging.getLogger(__name__)


# ============================================================
# QUIZ PROMPT
# The LLM is told to use ONLY the provided context — same
# constraint as the RAG prompt.
# ============================================================

def _build_quiz_prompt(context: str, num_questions: int) -> str:
    return f"""You are an AI quiz generator.

Create exactly {num_questions} multiple-choice questions from the
provided document context.

Return ONLY valid JSON. No markdown fences. No extra text.

Required format:

[
  {{
    "question": "Question text here",
    "options": [
      "Option A",
      "Option B",
      "Option C",
      "Option D"
    ],
    "answer": "Option A",
    "explanation": "Brief explanation of why this is correct."
  }}
]

Rules:
- Exactly 4 options per question.
- Exactly 1 correct answer per question.
- The "answer" field must exactly match one of the four options (word for word).
- Questions must be based ONLY on the provided document context.
- Do not use outside knowledge.
- Do not add markdown code fences.
- Do not add any text before or after the JSON array.
- Return JSON only.

DOCUMENT CONTEXT:

{context}
"""


# ============================================================
# STRIP MARKDOWN FENCES
# Gemini sometimes wraps JSON in ```json ... ``` despite being
# told not to. Strip them before parsing.
# ============================================================

def _strip_fences(raw: str) -> str:
    raw = re.sub(r"^```json\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"^```\s*",     "", raw)
    raw = re.sub(r"\s*```$",     "", raw)
    return raw.strip()


# ============================================================
# VALIDATE QUESTIONS
# Keep only questions that have the required fields and whose
# answer exactly matches one of the four options.
# ============================================================

def _validate_questions(raw_list: list) -> list:
    valid = []
    required_keys = {"question", "options", "answer", "explanation"}

    for item in raw_list:
        if not isinstance(item, dict):
            continue
        if not required_keys.issubset(item.keys()):
            logger.warning("Question skipped — missing keys: %s", item)
            continue
        if not isinstance(item["options"], list) or len(item["options"]) != 4:
            logger.warning("Question skipped — options count != 4: %s", item.get("question"))
            continue
        if item["answer"] not in item["options"]:
            logger.warning(
                "Question skipped — answer not in options: %s | %s",
                item["answer"], item["options"],
            )
            continue
        valid.append(item)

    return valid


# ============================================================
# GENERATE QUIZ  (public entry point)
# ============================================================

def generate_quiz(num_questions: int) -> list:
    """
    Generate MCQ questions from the indexed document.

    Steps:
      1. Retrieve the most relevant chunks from ChromaDB using a broad
         study-focused query (same approach as legacy_streamlit.py).
      2. Format the retrieved chunks into a context string.
      3. Ask the LLM to generate questions strictly from that context.
      4. Parse, validate, and return the question list.

    Returns:
        List of dicts:
        [
          {
            "question": "...",
            "options":  ["A", "B", "C", "D"],
            "answer":   "A",
            "explanation": "..."
          },
          ...
        ]

    Raises:
        ValueError  — if the LLM returns unparseable or invalid JSON.
        RuntimeError — propagated from retriever if ChromaDB is empty.
    """

    retriever = get_retriever()

    # Broad query to pull a varied set of content from the document.
    # This is identical to the approach in legacy_streamlit.py.
    documents = retriever.invoke(
        "Find important concepts, definitions, facts, examples, "
        "classifications and key points from the uploaded document."
    )

    if not documents:
        raise RuntimeError(
            "No content found in the indexed document. "
            "Please upload and index a PDF before generating a quiz."
        )

    context = format_context(documents)
    prompt  = _build_quiz_prompt(context, num_questions)

    logger.info("Generating quiz: %d questions requested.", num_questions)

    response = llm.invoke(prompt)
    raw      = _strip_fences(extract_text(response))

    logger.debug("Raw quiz response:\n%s", raw)

    # Parse JSON
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("JSON parse failed. Raw response:\n%s", raw)
        raise ValueError(
            f"The AI returned invalid JSON for the quiz. "
            f"Raw response (first 300 chars): {raw[:300]}"
        ) from exc

    if not isinstance(parsed, list):
        raise ValueError(
            f"Expected a JSON array of questions, got: {type(parsed).__name__}"
        )

    questions = _validate_questions(parsed)

    if not questions:
        raise ValueError(
            "The AI returned questions but none passed validation. "
            "Try again — the model may have formatted answers inconsistently."
        )

    logger.info("Quiz generated: %d valid questions.", len(questions))
    return questions
