"""Строит индекс базы знаний из data/docs и сохраняет его на диск.

Бэкенд выбирается переменной окружения RETRIEVER_BACKEND:
  - tfidf       (по умолчанию) — офлайн, без скачивания моделей
  - embeddings  — семантический поиск, при первом запуске качает модель

Запуск:
    python -m app.ingest
    RETRIEVER_BACKEND=embeddings python -m app.ingest          (Linux/macOS)
    set RETRIEVER_BACKEND=embeddings && python -m app.ingest   (Windows cmd)
"""
from pathlib import Path

from app.index_store import build_index, get_backend_name

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"


def main() -> None:
    backend = get_backend_name()
    print(f"Бэкенд ретривера: {backend}")
    print(f"Индексирую документы из {DOCS_DIR} ...")
    if backend == "embeddings":
        print("Первый запуск может занять время — скачивается модель эмбеддингов (~470 МБ).")

    index = build_index(DOCS_DIR, backend=backend)
    index.save()
    print(f"Готово: {len(index.chunks)} чанков из {len(list(DOCS_DIR.glob('*.md')))} документов.")


if __name__ == "__main__":
    main()
