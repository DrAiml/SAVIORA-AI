# AI Study Tutor

An AI-powered study and learning platform built with:

- **Frontend:** React
- **Backend:** FastAPI
- **AI:** LangChain + Google Gemini
- **Vector DB:** ChromaDB
- **PDF Parsing:** PyPDF / LangChain PyPDFLoader

---

## Project Structure

```
AI_Study_Tutor/
    backend/
        main.py                          # FastAPI app
        services/
            rag_service.py               # PDF loading, embedding, Q&A
            quiz_service.py              # Quiz generation
            recall_service.py            # Passage recall
            completion_service.py        # Sentence completion
        utils/
            config.py                    # LLM, embeddings, vector store
        uploads/                         # Uploaded PDFs (gitignored)
        chroma_db/                       # ChromaDB data (gitignored)
        requirements.txt
    frontend/                            # React app (Phase 7)
    legacy_streamlit.py                  # Original working Streamlit app
    .env                                 # Your API keys (never commit)
    .env.example                         # Template
    README.md
```

---

## Setup

### 1. Create your .env file

```
cp .env.example .env
```

Edit `.env` and add your Google API key:

```
GOOGLE_API_KEY=your_key_here
```

### 2. Install backend dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 3. Start the backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

---

## Testing — Phase 1 (RAG Service)

After starting the backend:

### Health check

```
GET http://localhost:8000/health
```

Expected:

```json
{"status": "ok"}
```

### Upload a PDF

```
POST http://localhost:8000/upload
Content-Type: multipart/form-data
file: <your PDF>
```

Expected:

```json
{"message": "Uploaded successfully", "filename": "document.pdf"}
```

### Index the PDF

```
POST http://localhost:8000/index
Content-Type: application/json

{"filename": "document.pdf"}
```

Expected:

```json
{"status": "indexed", "chunks": 42}
```

### Ask a question

```
POST http://localhost:8000/ask
Content-Type: application/json

{"question": "What is supervised learning?"}
```

Expected:

```json
{
  "answer": "...",
  "sources": [{"source": "document.pdf", "page": 3}]
}
```

---

## Common Errors

| Error | Cause | Fix |
|---|---|---|
| `GOOGLE_API_KEY not found` | Missing .env file | Create .env with your key |
| `No document has been indexed yet` | Ask before indexing | Upload and index a PDF first |
| `File not found in uploads/` | Wrong filename in /index | Check the exact filename from /upload response |
| ChromaDB permission error | Another process has the DB open | Stop other backend instances |

---

## Legacy Reference

`legacy_streamlit.py` is the original working Streamlit app.
To run it:

```bash
pip install streamlit
streamlit run legacy_streamlit.py
```

Do not modify this file — it is a working backup reference.
