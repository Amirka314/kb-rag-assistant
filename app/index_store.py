"""Построение и хранение поискового индекса по базе знаний.

Поддерживаются два бэкенда ретривера (выбор через переменную окружения
RETRIEVER_BACKEND, по умолчанию "tfidf"):

  - "tfidf"      — TF-IDF + косинусная близость. Работает полностью офлайн,
                     без скачивания моделей, детерминирован. Хорош как
                     надёжный дефолт и как база для объяснения на собеседовании.
  - "embeddings" — семантический поиск через sentence-transformers
                     (многоязычная модель). Понимает смысл, а не только
                     совпадение слов, но при первом запуске требует
                     интернет для скачивания весов модели (~470 МБ,
                     дальше берётся из локального кэша).

Оба бэкенда реализуют один и тот же интерфейс (build/save/load/search),
поэтому остальной код (rag.py, cli.py, web.py) от выбора бэкенда не зависит.
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path

from app.chunking import Chunk, load_chunks

INDEX_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


def get_backend_name() -> str:
    return os.getenv("RETRIEVER_BACKEND", "tfidf").strip().lower()


def _index_path(backend: str) -> Path:
    return INDEX_DIR / f"index.{backend}.pkl"


class KnowledgeBaseIndex:
    """TF-IDF ретривер (дефолтный бэкенд)."""

    def __init__(self, chunks: list[Chunk], vectorizer, matrix):
        self.chunks = chunks
        self.vectorizer = vectorizer
        self.matrix = matrix

    @classmethod
    def build(cls, docs_dir: Path) -> "KnowledgeBaseIndex":
        from sklearn.feature_extraction.text import TfidfVectorizer

        chunks = load_chunks(docs_dir)
        if not chunks:
            raise ValueError(f"В {docs_dir} не найдено ни одного .md документа с секциями '##'.")
        texts = [c.as_source_text() for c in chunks]
        vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)
        matrix = vectorizer.fit_transform(texts)
        return cls(chunks, vectorizer, matrix)

    def save(self, path: Path | None = None) -> None:
        path = path or _index_path("tfidf")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "vectorizer": self.vectorizer, "matrix": self.matrix}, f)

    @classmethod
    def load(cls, path: Path | None = None) -> "KnowledgeBaseIndex":
        path = path or _index_path("tfidf")
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(data["chunks"], data["vectorizer"], data["matrix"])

    def search(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.matrix)[0]
        ranked_idx = sims.argsort()[::-1][:top_k]
        return [RetrievedChunk(self.chunks[i], float(sims[i])) for i in ranked_idx if sims[i] > 0]


def build_index(docs_dir: Path, backend: str | None = None):
    """Фабрика: строит индекс нужным бэкендом."""
    backend = backend or get_backend_name()
    if backend == "embeddings":
        from app.embeddings_retriever import EmbeddingIndex

        return EmbeddingIndex.build(docs_dir)
    return KnowledgeBaseIndex.build(docs_dir)


def load_index(backend: str | None = None):
    """Фабрика: загружает уже построенный индекс нужным бэкендом."""
    backend = backend or get_backend_name()
    if backend == "embeddings":
        from app.embeddings_retriever import EmbeddingIndex

        return EmbeddingIndex.load()
    return KnowledgeBaseIndex.load()


def index_exists(backend: str | None = None) -> bool:
    backend = backend or get_backend_name()
    return _index_path(backend).exists()

