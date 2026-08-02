from __future__ import annotations

import unittest

from backend.core.generator import AnswerGenerator
from backend.core.grounding import GroundingScore, LexicalGroundingScorer
from backend.core.knowledge_base import KnowledgeChunk
from backend.core.pipeline import HallucinationPipeline
from backend.core.schemas import DetectRequest, QueryRequest


class _StubContradictionScorer(LexicalGroundingScorer):
    """Wraps the real lexical scorer but forces one specific claim to come
    back "contradicted", so pipeline-level contradiction handling (counts,
    hallucination_score, repair-loop routing) can be tested deterministically
    without depending on the real NLI model being installed.
    """

    method_name = "lexical"

    def __init__(self, contradict_claim_containing: str) -> None:
        self._trigger = contradict_claim_containing.lower()

    def score(self, claim, chunks):  # noqa: ANN001 - matches GroundingScorer protocol
        base = super().score(claim, chunks)
        if self._trigger in claim.lower():
            return GroundingScore(
                verdict="contradicted",
                support_score=base.support_score,
                chunk=base.chunk,
                method="lexical",
            )
        return base


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = HallucinationPipeline()

    def test_query_flow_flags_injected_unsupported_claim(self) -> None:
        result = self.pipeline.run_query(
            QueryRequest(
                question="What events appear in HelixCloud audit logs and how long are those logs retained?",
                top_k=4,
                simulate_hallucination=True,
            )
        )

        self.assertEqual(result.retrieval_mode, "lexical")
        self.assertGreaterEqual(result.unsupported_claim_count, 1)
        self.assertTrue(any(claim.verdict == "unsupported" for claim in result.claims))
        self.assertLess(result.answer_confidence, result.corrected_confidence)
        self.assertTrue(any(step.action in {"replaced", "removed"} for step in result.correction_steps))

    def test_detect_flow_keeps_grounded_answer(self) -> None:
        # Deliberately pinned to LexicalGroundingScorer rather than relying
        # on self.pipeline's env-derived default: this test's purpose is to
        # verify the repair loop leaves an already-correct, verbatim answer
        # untouched, not to assert that every scorer agrees on every claim.
        # NLI is measurably stricter than lexical overlap (see
        # eval/results/README.md) -- a pipeline test with an implicit,
        # environment-dependent scorer choice is exactly the kind of test
        # that looks like it's testing the pipeline but is actually testing
        # "whatever GROUNDING_MODE happens to be set to on this machine."
        pipeline = HallucinationPipeline(grounding_scorer=LexicalGroundingScorer())
        answer = (
            "Enterprise workspaces include audit logs for sign-in activity, source sync events, role changes, "
            "and report exports. Audit events are retained for 90 days."
        )
        result = pipeline.run_detection(
            DetectRequest(
                question="What events appear in HelixCloud audit logs and how long are those logs retained?",
                answer=answer,
                top_k=4,
            )
        )

        self.assertTrue(all(claim.verdict == "grounded" for claim in result.claims))
        self.assertEqual(result.corrected_answer, answer)

    def test_query_flow_reports_template_generation_when_no_api_key(self) -> None:
        result = self.pipeline.run_query(
            QueryRequest(
                question="What events appear in HelixCloud audit logs and how long are those logs retained?",
                top_k=4,
                simulate_hallucination=False,
            )
        )

        # No GEMINI_API_KEY is set in the test environment, so the default
        # pipeline (built with no explicit generator) must be honest about
        # falling back to the template path rather than silently pretending
        # to be an LLM answer.
        self.assertEqual(result.generation_mode, "template")
        self.assertIsNone(result.generation_warning)
        self.assertIn(result.grounding_mode, {"lexical", "nli"})
        self.assertTrue(all(claim.grounding_method == result.grounding_mode for claim in result.claims))

    def test_detect_flow_reports_user_provided_generation_mode(self) -> None:
        result = self.pipeline.run_detection(
            DetectRequest(
                question="What events appear in HelixCloud audit logs and how long are those logs retained?",
                answer="Audit events are retained for 90 days.",
                top_k=4,
            )
        )

        self.assertEqual(result.generation_mode, "user_provided")
        self.assertIsNone(result.generation_model)

    def test_contradicted_claims_are_counted_separately_from_unsupported(self) -> None:
        pipeline = HallucinationPipeline(
            generator=AnswerGenerator(api_key=None),
            grounding_scorer=_StubContradictionScorer(contradict_claim_containing="30 days"),
        )
        result = pipeline.run_detection(
            DetectRequest(
                question="How long are HelixCloud audit logs retained?",
                answer="Audit events are retained for 30 days.",
                top_k=4,
            )
        )

        self.assertEqual(result.contradicted_claim_count, 1)
        self.assertEqual(result.unsupported_claim_count, 0)
        self.assertEqual(result.grounded_claim_count, 0)
        # hallucination_score must include contradicted claims, not just
        # unsupported ones -- this is the bug the 3-way verdict introduced
        # if grounded_claim_count were still derived as `total - unsupported`.
        self.assertEqual(result.hallucination_score, 1.0)
        self.assertTrue(any(claim.verdict == "contradicted" for claim in result.claims))

    def test_recent_runs_tracks_latest_results(self) -> None:
        self.pipeline.run_query(
            QueryRequest(
                question="Which HelixCloud plans support single sign-on, and is email login still available?",
                top_k=4,
                simulate_hallucination=False,
            )
        )
        self.pipeline.run_detection(
            DetectRequest(
                question="What happens if a workspace token exceeds the HelixCloud API rate limit?",
                answer="The public API allows 120 requests per minute per workspace token.",
                top_k=4,
            )
        )

        history = self.pipeline.recent_runs(limit=5)
        summary = self.pipeline.analytics_summary()

        self.assertEqual(history.total_runs, 2)
        self.assertEqual(len(history.runs), 2)
        self.assertEqual(summary.total_runs, 2)
        self.assertIsNotNone(summary.latest_run)


if __name__ == "__main__":
    unittest.main()