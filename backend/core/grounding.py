from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Protocol, Sequence

from backend.core.knowledge_base import KnowledgeChunk
from backend.core.retriever import support_score as lexical_support_score

logger = logging.getLogger(__name__)

LEXICAL_GROUNDED_THRESHOLD = 0.42
NLI_ENTAILMENT_THRESHOLD = 0.55
NLI_CONTRADICTION_THRESHOLD = 0.55
DEFAULT_NLI_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

Verdict = str  # "grounded" | "unsupported" | "contradicted"


@dataclass(frozen=True)
class GroundingScore:
    verdict: Verdict
    support_score: float
    chunk: KnowledgeChunk
    method: str  # "lexical" | "nli"


class GroundingScorer(Protocol):
    method_name: str

    def score(self, claim: str, chunks: Sequence[KnowledgeChunk]) -> GroundingScore: ...


class LexicalGroundingScorer:
    """Token-overlap heuristic. Zero dependencies, always available, fast.

    Deliberately kept as the baseline/fallback rather than deleted: it's the
    thing the NLI scorer is benchmarked against in eval/run_eval.py, and it's
    what keeps the app fully functional with `pip install -r requirements.txt`
    alone (no ML deps). It cannot detect contradiction -- low word overlap
    just means "not obviously about the same thing", not "actively disagrees"
    -- so it only ever returns "grounded" or "unsupported".
    """

    method_name = "lexical"

    def score(self, claim: str, chunks: Sequence[KnowledgeChunk]) -> GroundingScore:
        if not chunks:
            raise ValueError("score() requires at least one candidate chunk")

        scored = [
            (chunk, lexical_support_score(claim, f"{chunk.title}. {chunk.text}"))
            for chunk in chunks
        ]
        best_chunk, best_score = max(scored, key=lambda item: item[1])
        verdict = "grounded" if best_score >= LEXICAL_GROUNDED_THRESHOLD else "unsupported"
        return GroundingScore(
            verdict=verdict,
            support_score=round(best_score, 4),
            chunk=best_chunk,
            method=self.method_name,
        )


class NLIGroundingScorer:
    """Cross-encoder NLI scorer.

    For each (claim, evidence) pair, predicts entailment / contradiction /
    neutral probabilities and derives a verdict from them, instead of just
    checking word overlap. This is what lets the detector distinguish "the
    evidence explicitly disagrees with this claim" (contradicted) from "the
    evidence just doesn't mention this" (unsupported) -- the standard
    intrinsic-vs-extrinsic hallucination distinction.

    Label order: sentence-transformers' own CrossEncoder NLI models (trained
    on SNLI/MultiNLI-style data) use the label order
    [contradiction, entailment, neutral]. This is asserted, not just assumed
    -- see `_assert_label_order` in eval/run_eval.py, which runs three
    unambiguous hand-labeled pairs through the loaded model before any real
    evaluation happens, specifically so a wrong label order fails loudly
    instead of silently producing inverted precision/recall numbers.
    """

    method_name = "nli"

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or os.getenv("NLI_MODEL_NAME", DEFAULT_NLI_MODEL_NAME)
        self._model = None
        self._load_failed = False

    def _load_model(self):
        if self._model is not None:
            return self._model
        if self._load_failed:
            return None
        try:
            from sentence_transformers import CrossEncoder
        except Exception:
            logger.warning("sentence-transformers not installed; NLI grounding unavailable.")
            self._load_failed = True
            return None
        try:
            self._model = CrossEncoder(self._model_name)
        except Exception:
            logger.exception("Failed to load NLI cross-encoder model '%s'.", self._model_name)
            self._load_failed = True
            return None
        return self._model

    @property
    def is_available(self) -> bool:
        return self._load_model() is not None

    def predict(self, claim: str, evidence_text: str) -> dict[str, float]:
        """Single (claim, evidence) prediction. Used by the pipeline for a
        single pair and directly by eval/run_eval.py, which evaluates the
        scorer against labeled (claim, evidence) pairs in isolation from
        retrieval.
        """
        model = self._load_model()
        if model is None:
            raise RuntimeError("NLI model is unavailable; install requirements-ml.txt")

        import numpy as np

        raw_scores = model.predict([(evidence_text, claim)])[0]
        exp_scores = np.exp(raw_scores - np.max(raw_scores))
        probs = exp_scores / exp_scores.sum()
        return {"contradiction": float(probs[0]), "entailment": float(probs[1]), "neutral": float(probs[2])}

    @staticmethod
    def _verdict_from_probs(contradiction: float, entailment: float) -> tuple[Verdict, float]:
        if entailment >= NLI_ENTAILMENT_THRESHOLD:
            return "grounded", entailment
        if contradiction >= NLI_CONTRADICTION_THRESHOLD:
            return "contradicted", contradiction
        return "unsupported", entailment

    def score(self, claim: str, chunks: Sequence[KnowledgeChunk]) -> GroundingScore:
        if not chunks:
            raise ValueError("score() requires at least one candidate chunk")
        model = self._load_model()
        if model is None:
            raise RuntimeError("NLI model is unavailable; install requirements-ml.txt")

        import numpy as np

        # Batch all candidate pairs into one model.predict() call rather than
        # looping (a cross-encoder forward pass per call is the expensive
        # part; batching keeps this to one inference call per claim).
        pairs = [(f"{chunk.title}. {chunk.text}", claim) for chunk in chunks]
        raw_scores = model.predict(pairs)
        exp_scores = np.exp(raw_scores - np.max(raw_scores, axis=1, keepdims=True))
        probs = exp_scores / exp_scores.sum(axis=1, keepdims=True)

        best_index = int(np.argmax(probs[:, 1]))  # highest entailment probability
        contradiction, entailment, _neutral = probs[best_index]
        verdict, confidence = self._verdict_from_probs(float(contradiction), float(entailment))
        return GroundingScore(
            verdict=verdict,
            support_score=round(confidence, 4),
            chunk=chunks[best_index],
            method=self.method_name,
        )


def build_default_grounding_scorer() -> GroundingScorer:
    """GROUNDING_MODE env var: lexical | nli | auto (default).
    "auto" prefers NLI and silently falls back to lexical if
    sentence-transformers / the model weights aren't available -- the same
    graceful-degradation pattern HybridRetriever already uses for embeddings.
    """
    mode = os.getenv("GROUNDING_MODE", "auto").lower()
    if mode == "lexical":
        return LexicalGroundingScorer()

    if mode in {"nli", "auto"}:
        nli_scorer = NLIGroundingScorer()
        if nli_scorer.is_available:
            return nli_scorer
        if mode == "nli":
            logger.warning("GROUNDING_MODE=nli requested but the model is unavailable; falling back to lexical.")

    return LexicalGroundingScorer()