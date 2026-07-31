from __future__ import annotations

import importlib.util
import unittest
import unittest.mock

from backend.core.grounding import (
    LexicalGroundingScorer,
    NLIGroundingScorer,
    build_default_grounding_scorer,
)
from backend.core.knowledge_base import KnowledgeChunk

SENTENCE_TRANSFORMERS_AVAILABLE = importlib.util.find_spec("sentence_transformers") is not None


def _chunk(chunk_id: str, text: str, title: str = "Doc") -> KnowledgeChunk:
    return KnowledgeChunk(chunk_id=chunk_id, source="helixcloud://docs/x", title=title, text=text)


class LexicalGroundingScorerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scorer = LexicalGroundingScorer()

    def test_matching_claim_is_grounded(self) -> None:
        chunks = [_chunk("chunk-001", "HelixCloud retains audit logs for 90 days.")]
        result = self.scorer.score("HelixCloud retains audit logs for 90 days.", chunks)

        self.assertEqual(result.verdict, "grounded")
        self.assertEqual(result.method, "lexical")
        self.assertEqual(result.chunk.chunk_id, "chunk-001")

    def test_unrelated_claim_is_unsupported(self) -> None:
        chunks = [_chunk("chunk-001", "HelixCloud retains audit logs for 90 days.")]
        result = self.scorer.score("The mitochondria is the powerhouse of the cell.", chunks)

        self.assertEqual(result.verdict, "unsupported")

    def test_raises_on_empty_candidate_list(self) -> None:
        with self.assertRaises(ValueError):
            self.scorer.score("Some claim.", [])

    def test_picks_best_of_multiple_candidates(self) -> None:
        chunks = [
            _chunk("chunk-001", "HelixCloud supports PDF and CSV uploads."),
            _chunk("chunk-002", "HelixCloud retains audit logs for 90 days."),
        ]
        result = self.scorer.score("HelixCloud retains audit logs for 90 days.", chunks)

        self.assertEqual(result.chunk.chunk_id, "chunk-002")
        self.assertEqual(result.verdict, "grounded")


class GroundingScorerFactoryTests(unittest.TestCase):
    def test_lexical_mode_env_returns_lexical_scorer(self) -> None:
        with unittest.mock.patch.dict("os.environ", {"GROUNDING_MODE": "lexical"}):
            scorer = build_default_grounding_scorer()
        self.assertEqual(scorer.method_name, "lexical")

    def test_auto_mode_falls_back_to_lexical_without_ml_deps(self) -> None:
        # This assertion only holds in an environment without
        # sentence-transformers installed (the lean core CI environment).
        # When requirements-ml.txt IS installed, "auto" should prefer NLI --
        # covered by NLIGroundingScorerTests below, which skips itself when
        # sentence-transformers is unavailable.
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            self.skipTest("sentence-transformers is installed; auto-mode fallback path not exercised here")
        with unittest.mock.patch.dict("os.environ", {"GROUNDING_MODE": "auto"}):
            scorer = build_default_grounding_scorer()
        self.assertEqual(scorer.method_name, "lexical")


@unittest.skipUnless(
    SENTENCE_TRANSFORMERS_AVAILABLE,
    "sentence-transformers not installed (pip install -r requirements-ml.txt). "
    "First run also downloads the cross-encoder/nli-deberta-v3-base weights "
    "from Hugging Face, which requires network access this sandbox does not have.",
)
class NLIGroundingScorerTests(unittest.TestCase):
    """These tests actually load the NLI model and run real inference.
    Skipped automatically when sentence-transformers isn't installed. Not
    run in this sandbox (huggingface.co is not in the network allowlist) --
    run these yourself with `pip install -r requirements-ml.txt` to verify.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.scorer = NLIGroundingScorer()

    def test_label_order_smoke_test(self) -> None:
        """The single most important test in this file: if the NLI model's
        label order were ever misread, this is what would catch it."""
        entailed = self.scorer.predict(
            claim="HelixCloud retains audit logs for 90 days.",
            evidence_text="HelixCloud retains audit logs for 90 days.",
        )
        contradicted = self.scorer.predict(
            claim="HelixCloud retains audit logs for 30 days.",
            evidence_text="HelixCloud retains audit logs for 90 days.",
        )
        neutral = self.scorer.predict(
            claim="HelixCloud supports PDF uploads.",
            evidence_text="HelixCloud retains audit logs for 90 days.",
        )

        self.assertGreater(entailed["entailment"], 0.5, entailed)
        self.assertGreater(contradicted["contradiction"], 0.5, contradicted)
        self.assertGreater(neutral["neutral"], entailed["neutral"], neutral)

    def test_score_picks_entailed_chunk_over_unrelated_chunk(self) -> None:
        chunks = [
            _chunk("chunk-001", "HelixCloud supports PDF and CSV uploads."),
            _chunk("chunk-002", "HelixCloud retains audit logs for 90 days."),
        ]
        result = self.scorer.score("HelixCloud retains audit logs for 90 days.", chunks)

        self.assertEqual(result.chunk.chunk_id, "chunk-002")
        self.assertEqual(result.verdict, "grounded")

    def test_score_detects_contradiction(self) -> None:
        chunks = [_chunk("chunk-001", "HelixCloud retains audit logs for 90 days.")]
        result = self.scorer.score("HelixCloud retains audit logs for 30 days.", chunks)

        self.assertEqual(result.verdict, "contradicted")


if __name__ == "__main__":
    unittest.main()