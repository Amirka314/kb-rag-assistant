"""Разбивка markdown-документов базы знаний на чанки для индексации.

Чанкуем по заголовкам второго уровня (##), потому что в базе знаний
поддержки именно такой раздел обычно соответствует одному законченному
ответу на вопрос пользователя. Это проще, чем чанкование по фиксированному
числу токенов, и даёт более "цельные" по смыслу куски текста.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    doc_id: str          # имя файла-источника
    doc_title: str        # заголовок документа (H1)
    section_title: str    # заголовок секции (H2), к которой относится чанк
    text: str              # текст чанка (без markdown-разметки заголовков)
    chunk_id: str          # уникальный идентификатор чанка

    def as_source_text(self) -> str:
        """Текст, который реально индексируется и показывается модели/пользователю."""
        return f"{self.doc_title} — {self.section_title}\n{self.text}".strip()


def _split_sections(markdown_text: str) -> list[tuple[str, str]]:
    """Возвращает список (заголовок_секции, текст_секции) по разбивке на '## '."""
    parts = re.split(r"\n##\s+", markdown_text)
    sections = []
    for part in parts[1:]:  # parts[0] — всё, что до первого "## " (обычно H1)
        lines = part.split("\n", 1)
        title = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        sections.append((title, body))
    return sections


def load_chunks(docs_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        h1_match = re.match(r"#\s+(.+)", raw)
        doc_title = h1_match.group(1).strip() if h1_match else path.stem

        for i, (section_title, body) in enumerate(_split_sections(raw)):
            if not body:
                continue
            chunks.append(
                Chunk(
                    doc_id=path.name,
                    doc_title=doc_title,
                    section_title=section_title,
                    text=body,
                    chunk_id=f"{path.stem}::{i}",
                )
            )
    return chunks
