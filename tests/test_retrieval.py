"""Базовые тесты ретривера: проверяем, что по осмысленному запросу
находится действительно релевантный документ, а не случайный.
"""
from pathlib import Path

import pytest

from app.index_store import KnowledgeBaseIndex

DOCS_DIR = Path(__file__).resolve().parent.parent / "data" / "docs"


@pytest.fixture(scope="module")
def index() -> KnowledgeBaseIndex:
    return KnowledgeBaseIndex.build(DOCS_DIR)


@pytest.mark.parametrize(
    "question, expected_doc_id",
    [
        ("что делать если платёж не прошёл", "oplata.md"),
        ("ошибка SYNC-409 конфликт версий", "oshibki-sinhronizacii.md"),
        ("как оформить возврат на карту", "vozvraty.md"),
        ("какие роли доступны сотрудникам", "roli-polzovateley.md"),
        ("лимиты запросов к API", "integracii.md"),
        ("требования к оборудованию для кассы", "ustanovka-i-obuchenie.md"),
    ],
)
def test_top_result_matches_expected_doc(index: KnowledgeBaseIndex, question: str, expected_doc_id: str) -> None:
    results = index.search(question, top_k=1)
    assert results, f"Пустой результат по запросу: {question!r}"
    assert results[0].chunk.doc_id == expected_doc_id


def test_irrelevant_query_returns_low_or_no_results(index: KnowledgeBaseIndex) -> None:
    results = index.search("погода в Алматы завтра", top_k=3)
    # Либо ничего не найдено, либо найденное имеет низкий score
    assert not results or all(r.score < 0.3 for r in results)
