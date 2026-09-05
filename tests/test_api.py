"""API-level tests hitting real FastAPI routes via TestClient.

Every existing test in tests/test_pipeline.py calls `HallucinationPipeline`
directly -- which means a broken route decorator, a wrong response_model,
missing dependency wiring in app.py, or a CORS/middleware regression would
pass the whole suite while the actual HTTP API was broken. These tests
exercise the real ASGI app end-to-end (request -> routing -> middleware ->
pipeline -> response serialization) to close that gap.

IMPORTANT: this module sets required environment variables *before*
importing backend.app, because backend/app.py builds its global `pipeline`
at import time (module-level `pipeline = HallucinationPipeline(...)`).
DATABASE_PATH=":memory:" keeps these tests from writing to a real
data/app.db file on disk.
"""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("DATABASE_PATH", ":memory:")
os.environ.setdefault("GROUNDING_MODE", "lexical")

from fastapi.testclient import TestClient

from backend.app import app, pipeline


class ApiHealthAndMetaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health_endpoint_returns_ok(self) -> None:
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_health_response_includes_request_id_header(self) -> None:
        response = self.client.get("/health")
        self.assertIn("x-request-id", response.headers)
        self.assertTrue(len(response.headers["x-request-id"]) > 0)

    def test_index_serves_frontend_html(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])

    def test_knowledge_base_endpoint_lists_documents(self) -> None:
        response = self.client.get("/knowledge-base")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("documents", body)
        self.assertIn("document_count", body)
        self.assertEqual(body["document_count"], len(body["documents"]))


class ApiQueryEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_query_endpoint_returns_full_response_contract(self) -> None:
        response = self.client.post(
            "/query",
            json={
                "question": "What events appear in HelixCloud audit logs?",
                "top_k": 4,
                "simulate_hallucination": False,
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        # Spot-check the response_model contract rather than every field --
        # the goal here is catching wiring/serialization breaks, not
        # re-testing pipeline logic already covered by test_pipeline.py.
        for field in (
            "run_id",
            "answer",
            "corrected_answer",
            "grounding_mode",
            "hallucination_score",
            "claims",
            "retrieved_chunks",
        ):
            self.assertIn(field, body)

    def test_query_endpoint_rejects_short_question(self) -> None:
        # Below QueryRequest's min_length=5 -- should fail FastAPI/Pydantic
        # validation (422), not reach the pipeline at all.
        response = self.client.post(
            "/query", json={"question": "hi", "top_k": 4, "simulate_hallucination": False}
        )
        self.assertEqual(response.status_code, 422)

    def test_query_endpoint_rejects_top_k_out_of_range(self) -> None:
        response = self.client.post(
            "/query",
            json={"question": "What is HelixCloud's audit log retention?", "top_k": 99},
        )
        self.assertEqual(response.status_code, 422)

    def test_query_endpoint_rejects_missing_question_field(self) -> None:
        response = self.client.post("/query", json={"top_k": 4})
        self.assertEqual(response.status_code, 422)


class ApiDetectEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_detect_endpoint_returns_full_response_contract(self) -> None:
        response = self.client.post(
            "/detect",
            json={
                "question": "What events appear in HelixCloud audit logs?",
                "answer": "HelixCloud audit logs capture login events and are retained for 90 days.",
                "top_k": 4,
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["generation_mode"], "user_provided")
        self.assertIn("correction_steps", body)

    def test_detect_endpoint_rejects_answer_too_short(self) -> None:
        response = self.client.post(
            "/detect",
            json={"question": "What is HelixCloud's retention policy?", "answer": "hi", "top_k": 4},
        )
        self.assertEqual(response.status_code, 422)


class ApiHistoryAndAnalyticsTests(unittest.TestCase):
    """Verifies /query and /detect calls are actually persisted and
    reflected through /runs/recent and /analytics/summary -- i.e. that the
    real HTTP layer is wired to the same pipeline/store, not a stray second
    instance.
    """

    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_query_then_recent_runs_reflects_it(self) -> None:
        before = pipeline.recent_runs(limit=50).total_runs

        response = self.client.post(
            "/query",
            json={
                "question": "Which HelixCloud plans support single sign-on?",
                "top_k": 4,
                "simulate_hallucination": False,
            },
        )
        self.assertEqual(response.status_code, 200)
        run_id = response.json()["run_id"]

        history = self.client.get("/runs/recent?limit=50")
        self.assertEqual(history.status_code, 200)
        body = history.json()
        self.assertEqual(body["total_runs"], before + 1)
        self.assertTrue(any(run["run_id"] == run_id for run in body["runs"]))

    def test_analytics_summary_endpoint_reflects_recorded_runs(self) -> None:
        self.client.post(
            "/detect",
            json={
                "question": "Does HelixCloud support single sign-on?",
                "answer": "HelixCloud supports SAML-based single sign-on on Business and Enterprise plans.",
                "top_k": 4,
            },
        )
        response = self.client.get("/analytics/summary")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertGreaterEqual(body["total_runs"], 1)
        self.assertIsNotNone(body["latest_run"])


class ApiErrorContractTests(unittest.TestCase):
    """Verifies the global exception handler (Step 4) is actually wired
    into the real app, not just correct in isolation.
    """

    def setUp(self) -> None:
        self.client = TestClient(app, raise_server_exceptions=False)

    def test_unhandled_pipeline_exception_returns_clean_error_contract(self) -> None:
        original_run_query = pipeline.run_query

        def _boom(payload):
            raise RuntimeError("simulated pipeline failure")

        pipeline.run_query = _boom  # type: ignore[method-assign]
        try:
            response = self.client.post(
                "/query",
                json={
                    "question": "What is HelixCloud's audit log retention?",
                    "top_k": 4,
                    "simulate_hallucination": False,
                },
            )
        finally:
            pipeline.run_query = original_run_query  # always restore, even on assertion failure

        self.assertEqual(response.status_code, 500)
        body = response.json()
        self.assertEqual(body["error"], "internal_server_error")
        self.assertIn("request_id", body)
        # The raw exception message/traceback must never reach the client.
        self.assertNotIn("simulated pipeline failure", response.text)
        self.assertNotIn("RuntimeError", response.text)


if __name__ == "__main__":
    unittest.main()