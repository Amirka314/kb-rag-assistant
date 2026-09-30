"""Интерактивный CLI для RAG-помощника по базе знаний.

Запуск:
    python -m app.cli
    RETRIEVER_BACKEND=embeddings python -m app.cli   (для семантического поиска)
"""
from app.index_store import get_backend_name, index_exists, load_index
from app.rag import RAGEngine


def main() -> None:
    backend = get_backend_name()
    if not index_exists(backend):
        print(f"Индекс для бэкенда '{backend}' не найден. Сначала запусти: python -m app.ingest")
        return

    index = load_index(backend)
    engine = RAGEngine(index)

    print(f"RAG-помощник по базе знаний продукта (бэкенд: {backend}).")
    print("Введите вопрос (или 'exit' для выхода).\n")
    while True:
        question = input("Вопрос: ").strip()
        if question.lower() in {"exit", "quit", "выход"}:
            break
        if not question:
            continue

        result = engine.answer(question)
        print(f"\nОтвет ({result.mode}):\n{result.answer}\n")
        if result.sources:
            print("Источники:")
            for r in result.sources:
                print(f"  - {r.chunk.doc_title} → {r.chunk.section_title} (score={r.score:.3f})")
        print()


if __name__ == "__main__":
    main()
