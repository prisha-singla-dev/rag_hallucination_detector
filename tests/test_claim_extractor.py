from __future__ import annotations

import unittest

from backend.core.claim_extractor import extract_claims


class ClaimExtractorTests(unittest.TestCase):
    def test_extract_claims_splits_and_normalizes_sentences(self) -> None:
        answer = (
            "Based on the retrieved evidence, RAG systems hallucinate when retrieval misses the right chunk. "
            "It also happens when the model answers beyond the evidence.\n"
            "- Tiny.\n"
            "Based on the retrieved evidence, RAG systems hallucinate when retrieval misses the right chunk."
        )

        claims = extract_claims(answer)

        self.assertEqual(
            claims,
            [
                "RAG systems hallucinate when retrieval misses the right chunk.",
                "It also happens when the model answers beyond the evidence.",
            ],
        )


if __name__ == "__main__":
    unittest.main()
