from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Sequence

from backend.core.knowledge_base import KnowledgeChunk


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")
NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")


@dataclass(frozen=True)
class RetrievalResult:
    chunk: KnowledgeChunk
    score: float


def tokenize(text: str) -> set[str]:
    return {match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)}


def lexical_score(query: str, text: str) -> float:
    query_tokens = tokenize(query)
    text_tokens = tokenize(text)
    if not query_tokens or not text_tokens:
        return 0.0

    overlap = query_tokens.intersection(text_tokens)
    coverage = len(overlap) / len(query_tokens)
    precision = len(overlap) / len(text_tokens)
    return round((0.7 * coverage) + (0.3 * precision), 4)


def numeric_alignment_score(query: str, text: str) -> float:
    query_numbers = set(NUMBER_PATTERN.findall(query))
    if not query_numbers:
        return 1.0

    text_numbers = set(NUMBER_PATTERN.findall(text))
    if not text_numbers:
        return 0.3

    shared = len(query_numbers.intersection(text_numbers))
    return round(shared / len(query_numbers), 4)


def support_score(query: str, text: str) -> float:
    lexical = lexical_score(query, text)
    numeric = numeric_alignment_score(query, text)
    return round((0.85 * lexical) + (0.15 * numeric), 4)


class HybridRetriever:
    def __init__(
        self,
        chunks: Sequence[KnowledgeChunk],
        embedding_model: str = "all-MiniLM-L6-v2",
        enable_embeddings: bool | None = None,
    ) -> None:
        self._chunks = list(chunks)
        self._embedding_model_name = embedding_model
        self._retrieval_mode = os.getenv("RETRIEVAL_MODE", "auto").lower()
        self._enable_embeddings = (
            os.getenv("ENABLE_SENTENCE_TRANSFORMERS", "0") == "1"
            if enable_embeddings is None
            else enable_embeddings
        )
        self._model = None
        self._chunk_embeddings: list[list[float]] | None = None
        self._faiss_index = None
        self._faiss_backend = None
        self._sentence_transformer_cls = None
        self._embedding_backend_checked = False
        self._faiss_backend_checked = False
        self._last_retrieval_mode = "lexical"

    @property
    def last_retrieval_mode(self) -> str:
        return self._last_retrieval_mode

    @property
    def supports_embeddings(self) -> bool:
        return self._enable_embeddings and self._load_embedding_backend() is not None

    def _load_embedding_backend(self):
        if self._embedding_backend_checked:
            return self._sentence_transformer_cls

        self._embedding_backend_checked = True
        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            self._sentence_transformer_cls = None
        else:
            self._sentence_transformer_cls = SentenceTransformer
        return self._sentence_transformer_cls

    def _load_faiss_backend(self):
        if self._faiss_backend_checked:
            return self._faiss_backend

        self._faiss_backend_checked = True
        try:
            import faiss
        except Exception:
            self._faiss_backend = None
        else:
            self._faiss_backend = faiss
        return self._faiss_backend

    def _ensure_embeddings(self) -> None:
        if not self.supports_embeddings or self._chunk_embeddings is not None:
            return

        sentence_transformer_cls = self._load_embedding_backend()
        assert sentence_transformer_cls is not None
        self._model = sentence_transformer_cls(self._embedding_model_name)
        embeddings = self._model.encode(
            [f"{chunk.title}. {chunk.text}" for chunk in self._chunks],
            normalize_embeddings=True,
        )
        self._chunk_embeddings = [embedding.tolist() for embedding in embeddings]

    def _ensure_faiss_index(self) -> None:
        if self._faiss_index is not None:
            return

        self._ensure_embeddings()
        faiss = self._load_faiss_backend()
        if faiss is None or self._chunk_embeddings is None:
            return

        import numpy as np

        embedding_matrix = np.array(self._chunk_embeddings, dtype="float32")
        index = faiss.IndexFlatIP(embedding_matrix.shape[1])
        index.add(embedding_matrix)
        self._faiss_index = index

    @staticmethod
    def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
        return sum(a * b for a, b in zip(left, right))

    def _embedding_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        self._ensure_embeddings()
        assert self._model is not None
        assert self._chunk_embeddings is not None

        query_embedding = self._model.encode([query], normalize_embeddings=True)[0].tolist()
        scored = [
            RetrievalResult(
                chunk=chunk,
                score=round(self._cosine_similarity(query_embedding, chunk_embedding), 4),
            )
            for chunk, chunk_embedding in zip(self._chunks, self._chunk_embeddings)
        ]
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

    def _faiss_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        self._ensure_faiss_index()
        assert self._model is not None
        assert self._faiss_index is not None

        import numpy as np

        query_embedding = self._model.encode([query], normalize_embeddings=True)
        scores, indices = self._faiss_index.search(np.array(query_embedding, dtype="float32"), top_k)
        return [
            RetrievalResult(chunk=self._chunks[index], score=round(float(score), 4))
            for score, index in zip(scores[0], indices[0])
            if index >= 0
        ]

    def _lexical_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        scored = [
            RetrievalResult(chunk=chunk, score=support_score(query, f"{chunk.title}. {chunk.text}"))
            for chunk in self._chunks
        ]
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

    def retrieve(self, query: str, top_k: int = 4) -> list[RetrievalResult]:
        if self.supports_embeddings and self._retrieval_mode in {"auto", "faiss"}:
            try:
                faiss_results = self._faiss_search(query, top_k)
                if faiss_results:
                    self._last_retrieval_mode = "faiss"
                    return faiss_results
            except Exception:
                pass

        if self.supports_embeddings and self._retrieval_mode in {"auto", "embedding"}:
            try:
                embedding_results = self._embedding_search(query, top_k)
                if embedding_results:
                    self._last_retrieval_mode = "embedding"
                    return embedding_results
            except Exception:
                pass

        self._last_retrieval_mode = "lexical"
        return self._lexical_search(query, top_k)

    def best_supporting_chunk(self, claim: str, chunks: Sequence[KnowledgeChunk]) -> RetrievalResult:
        chunk_list = list(chunks)
        if not chunk_list:
            raise ValueError("best_supporting_chunk requires at least one candidate chunk")

        scored = [
            RetrievalResult(chunk=chunk, score=support_score(claim, f"{chunk.title}. {chunk.text}"))
            for chunk in chunk_list
        ]
        return max(scored, key=lambda item: item.score, default=RetrievalResult(chunk=chunk_list[0], score=0.0))
