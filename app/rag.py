"""RAG-движок: находит релевантные фрагменты базы знаний и формирует ответ.

Ретривер (поиск фрагментов) подключается через фабрику index_store.load_index(),
поэтому этот файл не знает и не зависит от того, TF-IDF там или эмбеддинги —
см. app/index_store.py и app/embeddings_retriever.py.

Генерация ответа работает в двух режимах:
  - Если задан OPENAI_API_KEY — фрагменты передаются в модель как контекст,
    и она формулирует связный ответ со ссылкой на источник (классический RAG).
  - Если ключа нет — движок работает в "экстрактивном" режиме: возвращает
    сам найденный фрагмент как ответ. Так демо работает у любого, кто
    склонирует репозиторий, без необходимости платить за API.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

SYSTEM_PROMPT = (
    "Ты — ассистент поддержки, который отвечает на вопросы пользователей "
    "продукта, опираясь ТОЛЬКО на предоставленные фрагменты базы знаний. "
    "Если ответа во фрагментах нет — честно скажи, что не нашёл информацию, "
    "и предложи обратиться в поддержку. Отвечай кратко и по делу, на русском языке."
)


@dataclass
class RAGAnswer:
    answer: str
    sources: list
    mode: str  # "generative" или "extractive"


class RAGEngine:
    def __init__(self, index, top_k: int = 3):
        self.index = index
        self.top_k = top_k

    def answer(self, question: str) -> RAGAnswer:
        results = self.index.search(question, top_k=self.top_k)

        if not results:
            return RAGAnswer(
                answer="В базе знаний не нашлось ничего релевантного этому вопросу. "
                       "Рекомендую передать его в поддержку для уточнения.",
                sources=[],
                mode="extractive",
            )

        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return RAGAnswer(
                answer=self._generate_with_llm(question, results, api_key),
                sources=results,
                mode="generative",
            )

        return RAGAnswer(
            answer=self._extractive_answer(results),
            sources=results,
            mode="extractive",
        )

    @staticmethod
    def _extractive_answer(results) -> str:
        return results[0].chunk.text

    @staticmethod
    def _generate_with_llm(question: str, results, api_key: str) -> str:
        # Импорт внутри функции: пакет openai нужен только в generative-режиме,
        # чтобы extractive-режим работал без лишней зависимости.
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        context = "\n\n---\n\n".join(
            f"[Источник: {r.chunk.doc_title} — {r.chunk.section_title}]\n{r.chunk.text}"
            for r in results
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Контекст из базы знаний:\n{context}\n\nВопрос: {question}"},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
