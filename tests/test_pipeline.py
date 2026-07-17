from __future__ import annotations

import unittest

from backend.core.pipeline import HallucinationPipeline
from backend.core.schemas import DetectRequest, QueryRequest


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
        answer = (
            "Enterprise workspaces include audit logs for sign-in activity, source sync events, role changes, "
            "and report exports. Audit events are retained for 90 days."
        )
        result = self.pipeline.run_detection(
            DetectRequest(
                question="What events appear in HelixCloud audit logs and how long are those logs retained?",
                answer=answer,
                top_k=4,
            )
        )

        self.assertTrue(all(claim.verdict == "grounded" for claim in result.claims))
        self.assertEqual(result.corrected_answer, answer)

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
