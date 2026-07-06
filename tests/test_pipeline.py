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
                question="How can a RAG system detect unsupported claims and correct them?",
                top_k=4,
                simulate_hallucination=True,
            )
        )

        self.assertEqual(result.retrieval_mode, "lexical")
        self.assertTrue(any(claim.verdict == "unsupported" for claim in result.claims))
        self.assertLess(result.answer_confidence, result.corrected_confidence)

    def test_detect_flow_keeps_grounded_answer(self) -> None:
        answer = (
            "A production API should return the answer, a hallucination score, evidence used, "
            "claim-level verdicts, and a corrected answer when risk is high."
        )
        result = self.pipeline.run_detection(
            DetectRequest(
                question="What fields should a production hallucination detection API return?",
                answer=answer,
                top_k=4,
            )
        )

        self.assertTrue(all(claim.verdict == "grounded" for claim in result.claims))
        self.assertEqual(result.corrected_answer, answer)


if __name__ == "__main__":
    unittest.main()
