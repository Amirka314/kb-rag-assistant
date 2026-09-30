"""Семантический ретривер на эмбеддингах (sentence-transformers).

В отличие от TF-IDF, этот бэкенд понимает смысл вопроса, а не только
совпадение слов: вопрос "как вернуть деньги" найдёт документ про
"возврат средств", даже если ни одно слово не совпадает дословно.

Компромисс: при самом первом запуске (build или load) модель весом
~470 МБ скачивается с huggingface.co и кэшируется локально
(обычно в ~/.cache/huggingface). Дальше работает офлайн.

Модель: paraphrase-multilingual-MiniLM-L12-v2 — многоязычная, хорошо
работает с русским и казахским текстом, при этом достаточно компактная
для запуска на CPU без GPU.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.chunking import Chunk, load_chunks

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "index.embeddings.pkl"

_model_cache = None  # модель грузим один раз за процесс, а не на каждый вызов


def _get_model():
    global _model_cache
    if _model_cache is None:
        from sentence_transformers import SentenceTransformer

        _model_cache = SentenceTransformer(MODEL_NAME)
    return _model_cache


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


class EmbeddingIndex:
    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray):
        self.chunks = chunks
        self.embeddings = embeddings  # уже нормализованы (unit norm)

    @classmethod
    def build(cls, docs_dir: Path) -> "EmbeddingIndex":
        chunks = load_chunks(docs_dir)
        if not chunks:
            raise ValueError(f"В {docs_dir} не найдено ни одного .md документа с секциями '##'.")
        model = _get_model()
        texts = [c.as_source_text() for c in chunks]
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return cls(chunks, np.asarray(embeddings))

    def save(self, path: Path = INDEX_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "embeddings": self.embeddings}, f)

    @classmethod
    def load(cls, path: Path = INDEX_PATH) -> "EmbeddingIndex":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(data["chunks"], data["embeddings"])

    def search(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        model = _get_model()
        query_vec = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        # embeddings уже нормализованы -> скалярное произведение = косинусная близость
        sims = self.embeddings @ query_vec
        ranked_idx = np.argsort(sims)[::-1][:top_k]
        return [RetrievedChunk(self.chunks[i], float(sims[i])) for i in ranked_idx if sims[i] > 0.15]
