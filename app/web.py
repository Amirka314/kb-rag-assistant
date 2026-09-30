"""Веб-интерфейс поверх RAG-движка.

Запуск:
    uvicorn app.web:app --reload
Потом открой http://127.0.0.1:8000 в браузере.

Требует, чтобы индекс уже был построен:
    python -m app.ingest
"""
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.index_store import get_backend_name, index_exists, load_index
from app.rag import RAGEngine

app = FastAPI(title="kb-rag-assistant")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

_engine: RAGEngine | None = None


def get_engine() -> RAGEngine | None:
    global _engine
    if _engine is None:
        backend = get_backend_name()
        if not index_exists(backend):
            return None
        _engine = RAGEngine(load_index(backend))
    return _engine


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    engine = get_engine()
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "backend": get_backend_name(),
            "index_missing": engine is None,
            "question": "",
            "result": None,
        },
    )


@app.post("/ask", response_class=HTMLResponse)
def ask(request: Request, question: str = Form(...)):
    engine = get_engine()
    result = engine.answer(question) if engine else None
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "backend": get_backend_name(),
            "index_missing": engine is None,
            "question": question,
            "result": result,
        },
    )
