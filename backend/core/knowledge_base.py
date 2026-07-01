from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
KNOWLEDGE_BASE_PATH = DATA_DIR / "knowledge_base.json"


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    source: str
    title: str
    text: str


def load_knowledge_chunks() -> list[KnowledgeChunk]:
    with KNOWLEDGE_BASE_PATH.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    chunks: list[KnowledgeChunk] = []
    for index, record in enumerate(records, start=1):
        chunk_id = f"chunk-{index:03d}"
        chunks.append(
            KnowledgeChunk(
                chunk_id=chunk_id,
                source=record["source"],
                title=record["title"],
                text=record["text"].strip(),
            )
        )
    return chunks


def list_document_sources(chunks: Iterable[KnowledgeChunk]) -> list[dict[str, str]]:
    return [
        {
            "chunk_id": chunk.chunk_id,
            "title": chunk.title,
            "source": chunk.source,
        }
        for chunk in chunks
    ]
