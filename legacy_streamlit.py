import os
import json
import hashlib
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# ============================================================
# 1. BASIC CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error(
        """
        GOOGLE_API_KEY not found.

        Create a .env file in the same folder as this file:

        GOOGLE_API_KEY=your_api_key
        """
    )
    st.stop()


# ============================================================
# 2. DIRECTORIES
# ============================================================

UPLOAD_DIR = BASE_DIR / "uploaded_documents"
CHROMA_DIR = BASE_DIR / "chroma_db"

UPLOAD_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)


# ============================================================
# 3. STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Study Tutor",
    page_icon="🎓",
    layout="wide",
)


# ============================================================
# 4. LLM
# ============================================================

@st.cache_resource
def get_llm():

    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        temperature=0,
    )


# ============================================================
# 5. EMBEDDING MODEL
# ============================================================

@st.cache_resource
def get_embeddings():

    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2",
    )


# ============================================================
# 6. CHROMA VECTOR STORE
# ============================================================

@st.cache_resource
def get_vector_store():

    return Chroma(
        collection_name="study_tutor",
        persist_directory=str(CHROMA_DIR),
        embedding_function=get_embeddings(),
    )


# ============================================================
# 7. FILE HASH
# ============================================================

def get_file_hash(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        while True:

            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# 8. LOAD PDF
# ============================================================

def load_pdf(pdf_path):

    loader = PyPDFLoader(
        str(pdf_path)
    )

    documents = loader.load()

    return documents


# ============================================================
# 9. SPLIT DOCUMENT
# ============================================================

def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
    )

    chunks = splitter.split_documents(
        documents
    )

    return chunks


# ============================================================
# 10. PREPARE DOCUMENT METADATA
# ============================================================

def prepare_documents(
    chunks,
    filename,
    file_hash,
):

    documents = []

    for index, chunk in enumerate(chunks):

        metadata = dict(
            chunk.metadata
        )

        metadata["source"] = filename
        metadata["file_hash"] = file_hash
        metadata["chunk_id"] = (
            f"{file_hash}_{index}"
        )

        documents.append(
            Document(
                page_content=chunk.page_content,
                metadata=metadata,
            )
        )

    return documents


# ============================================================
# 11. CHECK IF PDF ALREADY EXISTS
# ============================================================

def is_already_indexed(
    vector_store,
    file_hash,
):

    try:

        result = vector_store.get(
            where={
                "file_hash": file_hash
            },
            limit=1,
        )

        ids = result.get("ids", [])

        return len(ids) > 0

    except Exception:

        return False


# ============================================================
# 12. INDEX ONE PDF
# ============================================================

def index_pdf(
    vector_store,
    pdf_path,
):

    filename = pdf_path.name

    file_hash = get_file_hash(
        pdf_path
    )

    # --------------------------------------------------------
    # Already indexed?
    # --------------------------------------------------------

    if is_already_indexed(
        vector_store,
        file_hash,
    ):

        return {
            "status": "existing",
            "chunks": 0,
        }


    # --------------------------------------------------------
    # Load PDF
    # --------------------------------------------------------

    documents = load_pdf(
        pdf_path
    )

    if not documents:

        return {
            "status": "empty",
            "chunks": 0,
        }


    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    chunks = split_documents(
        documents
    )

    if not chunks:

        return {
            "status": "empty",
            "chunks": 0,
        }


    # --------------------------------------------------------
    # Prepare metadata
    # --------------------------------------------------------

    prepared_documents = prepare_documents(
        chunks,
        filename,
        file_hash,
    )


    # --------------------------------------------------------
    # Add to Chroma
    # --------------------------------------------------------

    vector_store.add_documents(
        prepared_documents
    )


    return {
        "status": "indexed",
        "chunks": len(prepared_documents),
    }


# ============================================================
# 13. INDEX ALL PDF FILES
# ============================================================

def index_all_documents():

    vector_store = get_vector_store()

    pdf_files = list(
        UPLOAD_DIR.glob("*.pdf")
    )

    results = []

    for pdf_file in pdf_files:

        result = index_pdf(
            vector_store,
            pdf_file,
        )

        results.append(
            (
                pdf_file.name,
                result,
            )
        )

    return results


# ============================================================
# 14. RETRIEVER
# ============================================================

def get_retriever():

    vector_store = get_vector_store()

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 4
        },
    )


# ============================================================
# 15. FORMAT RETRIEVED DOCUMENTS
# ============================================================

def format_context(documents):

    if not documents:

        return "No relevant context found."


    context_parts = []

    for doc in documents:

        source = doc.metadata.get(
            "source",
            "Unknown",
        )

        page_number = doc.metadata.get(
            "page",
            0,
        )

        try:

            page_number = int(
                page_number
            ) + 1

        except Exception:

            page_number = 1


        context_parts.append(
            f"""
SOURCE: {source}
PAGE: {page_number}

{doc.page_content}
"""
        )

    return "\n\n---\n\n".join(
        context_parts
    )


# ============================================================
# 16. RAG PROMPT
# ============================================================

RAG_PROMPT = ChatPromptTemplate.from_template(
    """
You are an AI study assistant.

Answer the user's question ONLY using
the provided document context.

Rules:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not present in the
   document, clearly say:

   "I could not find this information
   in the uploaded document."

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
# 17. RAG CHAIN
# ============================================================

def build_rag_chain():

    retriever = get_retriever()

    llm = get_llm()

    rag_chain = (
        {
            "context":
                retriever
                | format_context,

            "question":
                RunnablePassthrough(),
        }

        | RAG_PROMPT

        | llm
    )

    return rag_chain


# ============================================================
# 18. EXTRACT RESPONSE TEXT
# ============================================================

def extract_text(response):

    content = response.content


    # Normal string
    if isinstance(
        content,
        str,
    ):

        return content


    # Gemini structured content
    if isinstance(
        content,
        list,
    ):

        text = ""

        for item in content:

            if isinstance(
                item,
                dict,
            ):

                if item.get(
                    "type"
                ) == "text":

                    text += item.get(
                        "text",
                        "",
                    )

        if text:

            return text


    return str(content)


# ============================================================
# 19. GENERATE QUIZ
# ============================================================

def generate_quiz(
    difficulty,
    number_of_questions,
):

    retriever = get_retriever()


    # Retrieve useful study context
    documents = retriever.invoke(
        """
        Find important concepts,
        definitions, facts, examples,
        classifications and key points
        from the uploaded document.
        """
    )


    context = format_context(
        documents
    )


    prompt = f"""
You are an AI quiz generator.

Create exactly {number_of_questions}
multiple-choice questions from the
provided document context.

Difficulty:
{difficulty}

Return ONLY valid JSON.

Required format:

[
  {{
    "question": "Question text",
    "options": [
      "Option A",
      "Option B",
      "Option C",
      "Option D"
    ],
    "answer": "Option A",
    "explanation": "Short explanation"
  }}
]

Rules:

- Exactly 4 options.
- Exactly 1 correct answer.
- The answer must exactly match
  one of the four options.
- Questions must be based ONLY on
  the provided context.
- Do not use outside knowledge.
- Do not add markdown.
- Do not add ```json.
- Return JSON only.

DOCUMENT CONTEXT:

{context}
"""


    response = get_llm().invoke(
        prompt
    )


    raw = extract_text(
        response
    ).strip()


    # --------------------------------------------------------
    # Remove accidental markdown fences
    # --------------------------------------------------------

    raw = re.sub(
        r"^```json\s*",
        "",
        raw,
        flags=re.IGNORECASE,
    )

    raw = re.sub(
        r"^```\s*",
        "",
        raw,
    )

    raw = re.sub(
        r"\s*```$",
        "",
        raw,
    )

    raw = raw.strip()


    # --------------------------------------------------------
    # Parse JSON
    # --------------------------------------------------------

    try:

        quiz = json.loads(
            raw
        )

    except json.JSONDecodeError:

        st.error(
            "Gemini returned invalid JSON."
        )

        st.code(
            raw,
            language="text",
        )

        return []


    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if not isinstance(
        quiz,
        list,
    ):

        st.error(
            "Quiz format is invalid."
        )

        return []


    valid_questions = []


    for question in quiz:

        if not isinstance(
            question,
            dict,
        ):

            continue


        if not all(
            key in question
            for key in [
                "question",
                "options",
                "answer",
                "explanation",
            ]
        ):

            continue


        if len(
            question["options"]
        ) != 4:

            continue


        if (
            question["answer"]
            not in question["options"]
        ):

            continue


        valid_questions.append(
            question
        )


    return valid_questions


# ============================================================
# 20. SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "quiz" not in st.session_state:

    st.session_state.quiz = []


if "quiz_index" not in st.session_state:

    st.session_state.quiz_index = 0


if "quiz_score" not in st.session_state:

    st.session_state.quiz_score = 0


if "quiz_submitted" not in st.session_state:

    st.session_state.quiz_submitted = False


if "quiz_result" not in st.session_state:

    st.session_state.quiz_result = None


# ============================================================
# 21. SIDEBAR
# ============================================================

with st.sidebar:

    st.title(
        "🎓 AI Study Tutor"
    )

    st.divider()


    # --------------------------------------------------------
    # Upload
    # --------------------------------------------------------

    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
    )


    if uploaded_files:

        for uploaded_file in uploaded_files:

            save_path = (
                UPLOAD_DIR
                / uploaded_file.name
            )


            with open(
                save_path,
                "wb",
            ) as file:

                file.write(
                    uploaded_file.getbuffer()
                )


        st.success(
            "PDF uploaded successfully."
        )


    # --------------------------------------------------------
    # Index
    # --------------------------------------------------------

    if st.button(
        "⚡ Index Documents",
        use_container_width=True,
    ):

        with st.spinner(
            "Reading, splitting and embedding documents..."
        ):

            try:

                results = (
                    index_all_documents()
                )


                if not results:

                    st.warning(
                        "No PDF documents found."
                    )


                for filename, result in results:

                    if (
                        result["status"]
                        == "indexed"
                    ):

                        st.success(
                            f"{filename}: "
                            f"{result['chunks']} "
                            f"chunks indexed."
                        )


                    elif (
                        result["status"]
                        == "existing"
                    ):

                        st.info(
                            f"{filename}: "
                            "already indexed."
                        )


                    else:

                        st.warning(
                            f"{filename}: "
                            "No readable text found."
                        )


            except Exception as e:

                st.error(
                    f"Indexing failed:\n\n{e}"
                )


    st.divider()


    # --------------------------------------------------------
    # Quiz settings
    # --------------------------------------------------------

    st.subheader(
        "Quiz Settings"
    )


    difficulty = st.selectbox(
        "Difficulty",
        [
            "Easy",
            "Medium",
            "Hard",
        ],
    )


    question_count = st.slider(
        "Questions",
        min_value=3,
        max_value=10,
        value=5,
    )


# ============================================================
# 22. MAIN UI
# ============================================================

st.title(
    "📚 AI Study Tutor"
)

st.caption(
    "Gemini Embedding 2 • ChromaDB • RAG • Gemini"
)


chat_tab, quiz_tab = st.tabs(
    [
        "💬 Ask PDF",
        "📝 Generate Quiz",
    ]
)


# ============================================================
# 23. ASK PDF TAB
# ============================================================

with chat_tab:

    # --------------------------------------------------------
    # Previous messages
    # --------------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    # --------------------------------------------------------
    # Question input
    # --------------------------------------------------------

    question = st.chat_input(
        "Ask something about your PDF..."
    )


    if question:

        # Save user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )


        with st.chat_message(
            "user"
        ):

            st.markdown(
                question
            )


        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Searching document..."
            ):

                try:

                    # Build RAG chain
                    rag_chain = (
                        build_rag_chain()
                    )


                    # Ask question
                    response = (
                        rag_chain.invoke(
                            question
                        )
                    )


                    answer = extract_text(
                        response
                    )


                    st.markdown(
                        answer
                    )


                    # ------------------------------------------------
                    # Show sources
                    # ------------------------------------------------

                    retrieved_docs = (
                        get_retriever()
                        .invoke(question)
                    )


                    if retrieved_docs:

                        st.divider()

                        st.caption(
                            "📄 Retrieved Sources"
                        )


                        seen_sources = set()


                        for doc in retrieved_docs:

                            source = (
                                doc.metadata.get(
                                    "source",
                                    "Unknown",
                                )
                            )


                            page = (
                                doc.metadata.get(
                                    "page",
                                    0,
                                )
                                + 1
                            )


                            source_key = (
                                source,
                                page,
                            )


                            if (
                                source_key
                                not in seen_sources
                            ):

                                st.write(
                                    f"• {source} "
                                    f"(Page {page})"
                                )


                                seen_sources.add(
                                    source_key
                                )


                except Exception as e:

                    answer = (
                        f"Error: {e}"
                    )

                    st.error(
                        answer
                    )


        # Save assistant response
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )


# ============================================================
# 24. QUIZ TAB
# ============================================================

with quiz_tab:

    st.header(
        "📝 Document Quiz"
    )


    # ========================================================
    # GENERATE QUIZ
    # ========================================================

    if st.button(
        "🚀 Generate Quiz",
        type="primary",
        key="generate_quiz_button",
    ):

        with st.spinner(
            "Generating quiz from your document..."
        ):

            try:

                quiz = generate_quiz(
                    difficulty,
                    question_count,
                )


                if quiz:

                    st.session_state.quiz = quiz

                    st.session_state.quiz_index = 0

                    st.session_state.quiz_score = 0

                    st.session_state.quiz_submitted = False

                    st.session_state.quiz_result = None

                    st.rerun()


                else:

                    st.warning(
                        "Could not generate a valid quiz."
                    )


            except Exception as e:

                st.error(
                    f"Quiz generation failed:\n\n{e}"
                )


    # ========================================================
    # DISPLAY QUIZ
    # ========================================================

    if st.session_state.quiz:

        index = (
            st.session_state.quiz_index
        )


        total = len(
            st.session_state.quiz
        )


        current_question = (
            st.session_state.quiz[index]
        )


        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        st.progress(
            (index + 1) / total
        )


        st.write(
            f"Question {index + 1} "
            f"of {total}"
        )


        # ----------------------------------------------------
        # Question
        # ----------------------------------------------------

        st.subheader(
            current_question["question"]
        )


        # ----------------------------------------------------
        # Answer options
        # ----------------------------------------------------

        selected_answer = st.radio(
            "Choose your answer:",
            current_question["options"],
            key=f"quiz_answer_{index}",
        )


        # ====================================================
        # BEFORE SUBMISSION
        # ====================================================

        if not st.session_state.quiz_submitted:

            if st.button(
                "✅ Submit Answer",
                key=f"submit_answer_{index}",
            ):

                # Check answer
                if (
                    selected_answer
                    == current_question["answer"]
                ):

                    st.session_state.quiz_score += 1

                    st.session_state.quiz_result = (
                        "correct"
                    )

                else:

                    st.session_state.quiz_result = (
                        "incorrect"
                    )


                # Mark answer submitted
                st.session_state.quiz_submitted = True


                # Rerun UI
                st.rerun()


        # ====================================================
        # AFTER SUBMISSION
        # ====================================================

        else:

            # ------------------------------------------------
            # Result
            # ------------------------------------------------

            if (
                st.session_state.quiz_result
                == "correct"
            ):

                st.success(
                    "🎉 Correct!"
                )

            else:

                st.error(
                    "❌ Incorrect!"
                )


                st.write(
                    f"**Correct answer:** "
                    f"{current_question['answer']}"
                )


            # ------------------------------------------------
            # Explanation
            # ------------------------------------------------

            st.info(
                "💡 "
                + current_question["explanation"]
            )


            # =================================================
            # NEXT QUESTION
            # =================================================

            if index < total - 1:

                if st.button(
                    "➡️ Next Question",
                    type="primary",
                    key=f"next_question_{index}",
                ):

                    st.session_state.quiz_index += 1

                    st.session_state.quiz_submitted = False

                    st.session_state.quiz_result = None


                    # IMPORTANT
                    # Move to next question
                    st.rerun()


            # =================================================
            # QUIZ FINISHED
            # =================================================

            else:

                score = (
                    st.session_state.quiz_score
                )


                st.success(
                    f"🏆 Quiz Complete!"
                )


                st.metric(
                    "Your Score",
                    f"{score}/{total}",
                )


                if st.button(
                    "🔄 Generate New Quiz",
                    key="generate_new_quiz",
                ):

                    st.session_state.quiz = []

                    st.session_state.quiz_index = 0

                    st.session_state.quiz_score = 0

                    st.session_state.quiz_submitted = False

                    st.session_state.quiz_result = None


                    st.rerun()