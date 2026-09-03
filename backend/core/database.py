from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from backend.core.schemas import AnalyticsSummaryResponse, RunRecord


# Free-tier hosts restart containers on every deploy (and sometimes on
# idle-sleep wake), which wiped the old in-memory deque completely. SQLite
# gives us a durable file that survives restarts with zero extra
# infrastructure. RUN_HISTORY_LIMIT is a soft cap (pruned after each insert)
# so disk usage stays bounded on small hosting tiers -- 5,000 rows of this
# schema is a few MB at most, comfortably higher than the old 50-row cap
# since disk is cheap where the old in-process memory budget wasn't.
RUN_HISTORY_LIMIT = 5000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    mode TEXT NOT NULL,
    question TEXT NOT NULL,
    hallucination_score REAL NOT NULL,
    answer_confidence REAL NOT NULL,
    corrected_confidence REAL NOT NULL,
    unsupported_claim_count INTEGER NOT NULL,
    contradicted_claim_count INTEGER NOT NULL,
    retrieval_mode TEXT NOT NULL,
    grounding_mode TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_id_desc ON runs (id DESC);
"""


class RunHistoryStore:
    """Thread-safe SQLite-backed store for run history and analytics.

    A single connection is held open for the lifetime of the store (rather
    than opening one per call) because the default `:memory:` mode -- used
    when no db_path is given, e.g. in tests -- would otherwise lose all data
    between calls; each new connection to ':memory:' is a distinct empty
    database. A lock serializes access since FastAPI's sync routes run in a
    threadpool and sqlite3 connections aren't safe for concurrent use from
    multiple threads.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._lock = threading.Lock()
        target = db_path or ":memory:"
        if target != ":memory:":
            Path(target).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(target, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            if target != ":memory:":
                # WAL improves concurrent read/write behavior for a
                # file-backed db under FastAPI's threadpool; irrelevant
                # (and unsupported in a meaningful way) for :memory:.
                self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.commit()

    def add_run(self, record: RunRecord) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO runs (
                    run_id, created_at, mode, question, hallucination_score,
                    answer_confidence, corrected_confidence,
                    unsupported_claim_count, contradicted_claim_count,
                    retrieval_mode, grounding_mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.run_id,
                    record.created_at.isoformat(),
                    record.mode,
                    record.question,
                    record.hallucination_score,
                    record.answer_confidence,
                    record.corrected_confidence,
                    record.unsupported_claim_count,
                    record.contradicted_claim_count,
                    record.retrieval_mode,
                    record.grounding_mode,
                ),
            )
            # Soft cap: prune oldest rows beyond RUN_HISTORY_LIMIT so disk
            # usage stays bounded on small hosting tiers.
            self._conn.execute(
                """
                DELETE FROM runs WHERE id NOT IN (
                    SELECT id FROM runs ORDER BY id DESC LIMIT ?
                )
                """,
                (RUN_HISTORY_LIMIT,),
            )
            self._conn.commit()

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            run_id=row["run_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            mode=row["mode"],
            question=row["question"],
            hallucination_score=row["hallucination_score"],
            answer_confidence=row["answer_confidence"],
            corrected_confidence=row["corrected_confidence"],
            unsupported_claim_count=row["unsupported_claim_count"],
            contradicted_claim_count=row["contradicted_claim_count"],
            retrieval_mode=row["retrieval_mode"],
            grounding_mode=row["grounding_mode"],
        )

    def recent(self, limit: int = 10) -> list[RunRecord]:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)
            )
            rows = cursor.fetchall()
        return [self._row_to_record(row) for row in rows]

    def total_count(self) -> int:
        with self._lock:
            cursor = self._conn.execute("SELECT COUNT(*) AS n FROM runs")
            return int(cursor.fetchone()["n"])

    def summary(self) -> AnalyticsSummaryResponse:
        with self._lock:
            totals_row = self._conn.execute(
                """
                SELECT
                    COUNT(*) AS total_runs,
                    SUM(CASE WHEN mode = 'query' THEN 1 ELSE 0 END) AS query_runs,
                    AVG(hallucination_score) AS avg_hallucination_score,
                    AVG(corrected_confidence - answer_confidence) AS avg_confidence_delta
                FROM runs
                """
            ).fetchone()
            latest_row = self._conn.execute(
                "SELECT * FROM runs ORDER BY id DESC LIMIT 1"
            ).fetchone()

        total_runs = int(totals_row["total_runs"] or 0)
        if total_runs == 0:
            return AnalyticsSummaryResponse(
                total_runs=0,
                query_runs=0,
                detect_runs=0,
                avg_hallucination_score=0.0,
                avg_confidence_delta=0.0,
                latest_run=None,
            )

        query_runs = int(totals_row["query_runs"] or 0)
        return AnalyticsSummaryResponse(
            total_runs=total_runs,
            query_runs=query_runs,
            detect_runs=total_runs - query_runs,
            avg_hallucination_score=round(float(totals_row["avg_hallucination_score"] or 0.0), 4),
            avg_confidence_delta=round(float(totals_row["avg_confidence_delta"] or 0.0), 4),
            latest_run=self._row_to_record(latest_row) if latest_row else None,
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()