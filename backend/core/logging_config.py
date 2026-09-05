from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any


# Plain-text logs are hard to search/aggregate once this is deployed
# anywhere with a real log viewer (Render, Railway, or a future
# ELK/Datadog pipeline) -- JSON lines are greppable and parseable without a
# custom parser. LOG_LEVEL is env-overridable so production can run at INFO
# while local dev can drop to DEBUG without a code change.
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED_LOGRECORD_KEYS:
                continue
            payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


_RESERVED_LOGRECORD_KEYS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()) | {
    "message",
    "asctime",
}


def configure_logging(level: str | None = None) -> None:
    """Idempotent logging setup: call once at app startup."""
    root = logging.getLogger()
    if getattr(root, "_json_configured", False):
        return
    root.setLevel(level or "INFO")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.handlers = [handler]
    root._json_configured = True  # type: ignore[attr-defined]


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def now_ms() -> float:
    return time.perf_counter() * 1000