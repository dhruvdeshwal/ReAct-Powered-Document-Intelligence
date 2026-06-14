"""
main.py
-------
FastAPI backend for the ReAct Document Analysis System.

Endpoints:
  POST /upload      -> upload a document (PDF/DOCX/TXT), process & index it
  POST /query       -> ask a question, run the ReAct agent, return answer + trace
  GET  /documents   -> list indexed documents
  DELETE /documents/{filename} -> remove a document from the index
  GET  /status      -> health/status info
"""

import os
import shutil
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from document_processor import chunk_document
from vector_store import VectorStore
from agent import ReActAgent

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

app = FastAPI(title="ReAct Document Analysis API")

# Allow Streamlit (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize shared resources once at startup
vector_store = VectorStore()
agent = ReActAgent(vector_store)


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str
    use_decomposition: bool = True


class QueryResponse(BaseModel):
    answer: str
    trace: list
    sub_questions: list[str] | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/status")
def status():
    """Basic health/status info."""
    return {
        "status": "ok",
        "indexed_documents": vector_store.list_documents(),
        "total_chunks": vector_store.count(),
        "llm_model": os.getenv("LLM_MODEL"),
        "embedding_model": os.getenv("EMBEDDING_MODEL"),
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document, process it into chunks, and add to the vector store."""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        chunks = chunk_document(save_path)
        added = vector_store.add_chunks(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")

    return {
        "filename": file.filename,
        "chunks_added": added,
        "total_chunks": vector_store.count(),
    }


@app.get("/documents")
def list_documents():
    """List all indexed documents."""
    return {"documents": vector_store.list_documents()}


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    """Remove a document and its chunks from the index (and disk)."""
    vector_store.delete_document(filename)

    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    return {"deleted": filename}

@app.get("/")
def root():
    return {
        "message": "ReAct Document Analysis API running"
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """Run the ReAct agent against the indexed documents to answer a question."""
    if vector_store.count() == 0:
        raise HTTPException(status_code=400, detail="No documents indexed yet. Upload a document first.")

    try:
        if request.use_decomposition:
            result = agent.run_with_decomposition(request.question)
        else:
            result = agent.run(request.question)
    except RuntimeError as e:
        # e.g. missing GROQ_API_KEY
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    return {
        "answer": result["answer"],
        "trace": result.get("trace", []),
        "sub_questions": result.get("sub_questions"),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)