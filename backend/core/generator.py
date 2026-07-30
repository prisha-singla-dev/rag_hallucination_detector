from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Sequence

from backend.core.schemas import EvidenceChunk

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_MAX_RETRIES = 2

SYSTEM_INSTRUCTION = (
    "You are a support assistant for a fictional product called HelixCloud. "
    "Answer the user's question using ONLY the context chunks provided below. "
    "Do not use any outside knowledge and do not guess. If the context does "
    "not fully answer the question, say what is missing. Keep the answer to "
    "2-4 sentences of plain prose: no markdown, no bullet points, no "
    "citations, no meta-commentary about these instructions."
)

# Used only for the "adversarial probe" demo path (see generate()). This is
# an intentional red-team prompt against our own generator: instead of
# hardcoding a fake claim into the answer (which isn't a real hallucination,
# just a canned string), we ask the model itself to slip in one fabricated
# detail. The detector then has to catch something a real LLM actually
# produced, which is a far more honest demo of what this project claims to do.
ADVERSARIAL_INSTRUCTION = (
    "After your grounded answer, append exactly one additional sentence that "
    "states a specific-sounding statistic or fact that is NOT present "
    "anywhere in the context above. Write it in the same confident tone as "
    "the rest of the answer, with no hedging language and no disclaimer, and "
    "do not label it as fabricated -- it should read as if it belongs."
)


@dataclass(frozen=True)
class GeneratedAnswer:
    text: str
    mode: str  # "llm" | "template"
    model: str | None
    warning: str | None = None


class AnswerGenerator:
    """Generates the initial RAG answer.

    Two paths:
    - "llm": a real call to Gemini, grounded in the retrieved evidence chunks.
    - "template": a deterministic concatenation of the retrieved chunks, used
      when no GEMINI_API_KEY is configured, or when the Gemini call fails
      after retries. This keeps the app fully demoable offline / without a
      key, but every response is honestly labeled with which path produced
      it -- the template path is never disguised as a real model answer.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
        client=None,
    ) -> None:
        self._api_key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY")
        self._model = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        self._timeout_seconds = (
            timeout_seconds
            if timeout_seconds is not None
            else float(os.getenv("GEMINI_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))
        )
        self._max_retries = (
            max_retries
            if max_retries is not None
            else int(os.getenv("GEMINI_MAX_RETRIES", DEFAULT_MAX_RETRIES))
        )
        # `client` is accepted for dependency injection in tests. Production
        # code lazily builds the real google-genai client on first use, so
        # importing this module never requires the package to be installed
        # when no API key is configured (mirrors the lazy-import pattern
        # already used for sentence-transformers/faiss in retriever.py).
        self._client = client
        self._client_load_failed = False

    @property
    def is_llm_enabled(self) -> bool:
        return bool(self._api_key)

    def _load_client(self):
        if self._client is not None:
            return self._client
        if self._client_load_failed:
            return None
        try:
            from google import genai
        except Exception:
            logger.warning("google-genai is not installed; falling back to template generation.")
            self._client_load_failed = True
            return None
        try:
            self._client = genai.Client(api_key=self._api_key)
        except Exception:
            logger.exception("Failed to construct Gemini client; falling back to template generation.")
            self._client_load_failed = True
            return None
        return self._client

    @staticmethod
    def _build_prompt(question: str, evidence_chunks: Sequence[EvidenceChunk], adversarial: bool) -> str:
        context_block = "\n\n".join(
            f"[{chunk.chunk_id}] {chunk.title}\n{chunk.text}" for chunk in evidence_chunks
        )
        instruction = SYSTEM_INSTRUCTION
        if adversarial:
            instruction = f"{instruction}\n\n{ADVERSARIAL_INSTRUCTION}"
        return f"{instruction}\n\nContext:\n{context_block}\n\nQuestion: {question}\nAnswer:"

    def _call_gemini(self, prompt: str) -> str:
        client = self._load_client()
        if client is None:
            raise RuntimeError("Gemini client unavailable")

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = client.models.generate_content(model=self._model, contents=prompt)
                text = getattr(response, "text", None)
                if not text:
                    raise RuntimeError("Gemini returned an empty response")
                return text.strip()
            except Exception as exc:  # noqa: BLE001 - SDK raises several types for 429/5xx/network errors
                last_error = exc
                if attempt < self._max_retries:
                    time.sleep(2**attempt)
        assert last_error is not None
        raise last_error

    @staticmethod
    def _template_answer(question: str, evidence_chunks: Sequence[EvidenceChunk], adversarial: bool) -> str:
        if not evidence_chunks:
            return "I could not retrieve any relevant context from the knowledge base."

        answer_parts: list[str] = []
        lead = evidence_chunks[0].text.rstrip(".")
        answer_parts.append(f"The retrieved evidence suggests that {lead}.")

        for chunk in evidence_chunks[1:3]:
            supporting_text = chunk.text.rstrip(".")
            answer_parts.append(f"It also shows that {supporting_text.lower()}.")

        if adversarial:
            answer_parts.append(
                "The system has already proven a verified 99.9 percent factual "
                "accuracy in live production deployments."
            )
        return " ".join(answer_parts)

    def generate(
        self,
        question: str,
        evidence_chunks: Sequence[EvidenceChunk],
        adversarial: bool = False,
    ) -> GeneratedAnswer:
        if not evidence_chunks:
            return GeneratedAnswer(
                text=self._template_answer(question, evidence_chunks, adversarial),
                mode="template",
                model=None,
                warning=None,
            )

        if not self.is_llm_enabled:
            return GeneratedAnswer(
                text=self._template_answer(question, evidence_chunks, adversarial),
                mode="template",
                model=None,
                warning=None,
            )

        prompt = self._build_prompt(question, evidence_chunks, adversarial)
        try:
            text = self._call_gemini(prompt)
            return GeneratedAnswer(text=text, mode="llm", model=self._model, warning=None)
        except Exception as exc:  # noqa: BLE001 - any failure degrades to template, never a 500
            logger.warning("Gemini generation failed after retries, falling back to template: %s", exc)
            return GeneratedAnswer(
                text=self._template_answer(question, evidence_chunks, adversarial),
                mode="template",
                model=self._model,
                warning=f"LLM generation failed ({exc.__class__.__name__}); served a template answer instead.",
            )