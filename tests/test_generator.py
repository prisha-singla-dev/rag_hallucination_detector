from __future__ import annotations

import unittest
from unittest.mock import patch

from backend.core.generator import AnswerGenerator
from backend.core.schemas import EvidenceChunk


def _chunk(chunk_id: str, text: str) -> EvidenceChunk:
    return EvidenceChunk(chunk_id=chunk_id, title="Doc", source="helixcloud://docs/x", text=text, score=0.9)


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeModels:
    def __init__(self, responses: list) -> None:
        self._responses = responses
        self.calls: list[dict] = []

    def generate_content(self, model: str, contents: str):
        self.calls.append({"model": model, "contents": contents})
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return _FakeResponse(response)


class _FakeClient:
    def __init__(self, responses: list) -> None:
        self.models = _FakeModels(responses)


class GeneratorTemplateFallbackTests(unittest.TestCase):
    def test_no_api_key_uses_template_mode(self) -> None:
        generator = AnswerGenerator(api_key=None)
        chunks = [_chunk("chunk-001", "HelixCloud retains audit logs for 90 days.")]

        result = generator.generate("How long are audit logs kept?", chunks, adversarial=False)

        self.assertEqual(result.mode, "template")
        self.assertIsNone(result.model)
        self.assertIsNone(result.warning)
        self.assertIn("90 days", result.text)

    def test_no_evidence_short_circuits_without_calling_llm(self) -> None:
        client = _FakeClient(responses=["should not be called"])
        generator = AnswerGenerator(api_key="fake-key", client=client)

        result = generator.generate("Unanswerable question?", [], adversarial=False)

        self.assertEqual(result.mode, "template")
        self.assertEqual(client.models.calls, [])

    def test_adversarial_template_injects_fabricated_sentence(self) -> None:
        generator = AnswerGenerator(api_key=None)
        chunks = [_chunk("chunk-001", "HelixCloud retains audit logs for 90 days.")]

        result = generator.generate("How long are audit logs kept?", chunks, adversarial=True)

        self.assertIn("99.9 percent", result.text)


class GeneratorLLMPathTests(unittest.TestCase):
    """Exercises the Gemini code path with a fake client so no network call
    is made. This cannot prove the real API integration works end to end --
    that requires a live GEMINI_API_KEY -- but it does prove the request is
    built correctly, the response is parsed correctly, and retry/fallback
    behavior is correct.
    """

    def test_successful_call_returns_llm_mode(self) -> None:
        client = _FakeClient(responses=["HelixCloud retains audit logs for 90 days."])
        generator = AnswerGenerator(api_key="fake-key", model="gemini-2.5-flash", client=client)
        chunks = [_chunk("chunk-001", "HelixCloud retains audit logs for 90 days.")]

        result = generator.generate("How long are audit logs kept?", chunks, adversarial=False)

        self.assertEqual(result.mode, "llm")
        self.assertEqual(result.model, "gemini-2.5-flash")
        self.assertIsNone(result.warning)
        self.assertEqual(len(client.models.calls), 1)
        self.assertIn("How long are audit logs kept?", client.models.calls[0]["contents"])

    @patch("backend.core.generator.time.sleep")
    def test_retries_then_falls_back_to_template_on_persistent_failure(self, mock_sleep) -> None:
        client = _FakeClient(responses=[RuntimeError("429"), RuntimeError("429"), RuntimeError("429")])
        generator = AnswerGenerator(api_key="fake-key", client=client, max_retries=2, timeout_seconds=1)
        chunks = [_chunk("chunk-001", "HelixCloud retains audit logs for 90 days.")]

        result = generator.generate("How long are audit logs kept?", chunks, adversarial=False)

        self.assertEqual(result.mode, "template")
        self.assertIsNotNone(result.warning)
        self.assertIn("RuntimeError", result.warning)
        # max_retries=2 means 3 total attempts (1 initial + 2 retries).
        self.assertEqual(len(client.models.calls), 3)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("backend.core.generator.time.sleep")
    def test_succeeds_after_one_retry(self, mock_sleep) -> None:
        client = _FakeClient(responses=[RuntimeError("transient"), "Recovered answer."])
        generator = AnswerGenerator(api_key="fake-key", client=client, max_retries=2, timeout_seconds=1)
        chunks = [_chunk("chunk-001", "HelixCloud retains audit logs for 90 days.")]

        result = generator.generate("How long are audit logs kept?", chunks, adversarial=False)

        self.assertEqual(result.mode, "llm")
        self.assertEqual(result.text, "Recovered answer.")
        self.assertEqual(len(client.models.calls), 2)
        self.assertEqual(mock_sleep.call_count, 1)


if __name__ == "__main__":
    unittest.main()