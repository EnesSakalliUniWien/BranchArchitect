from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterator

import pytest
from flask import Flask

from webapp.config import Config
from webapp.services.logging import config as logging_config
from webapp.services.logging.config import configure_logging

_OWNED_HANDLER_ATTR = "_phylo_movies_handler"


@pytest.fixture(autouse=True)
def cleanup_backend_logging() -> Iterator[None]:
    root_logger = logging.getLogger()
    original_level = root_logger.level
    yield
    for handler in list(root_logger.handlers):
        if getattr(handler, _OWNED_HANDLER_ATTR, False):
            root_logger.removeHandler(handler)
            handler.close()
    root_logger.setLevel(original_level)


def test_configure_logging_honors_configured_log_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Config, "LOG_LEVEL", "WARNING")
    monkeypatch.setattr(Config, "LOG_FORMAT", "text")
    monkeypatch.setattr(Config, "LOG_FILE", None)
    monkeypatch.setattr(logging_config.sys, "frozen", False, raising=False)

    app = Flask("logging-level-test")
    configure_logging(app)

    owned_handlers = _owned_root_handlers()

    assert logging.getLogger().level == logging.WARNING
    assert app.logger.handlers == []
    assert app.logger.propagate is True
    assert len(owned_handlers) == 1
    assert owned_handlers[0].level == logging.WARNING
    assert not any(isinstance(handler, RotatingFileHandler) for handler in owned_handlers)


def test_configure_logging_writes_optional_file_sink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_path = tmp_path / "backend.log"
    monkeypatch.setattr(Config, "LOG_LEVEL", "INFO")
    monkeypatch.setattr(Config, "LOG_FORMAT", "text")
    monkeypatch.setattr(Config, "LOG_FILE", str(log_path))

    app = Flask("logging-file-test")
    configure_logging(app)

    app.logger.info("optional file sink check")
    _flush_owned_handlers()

    assert "optional file sink check" in log_path.read_text(encoding="utf-8")


def test_json_request_logs_include_request_metadata_and_skip_healthchecks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log_path = tmp_path / "backend.jsonl"
    monkeypatch.setattr(Config, "LOG_LEVEL", "INFO")
    monkeypatch.setattr(Config, "LOG_FORMAT", "json")
    monkeypatch.setattr(Config, "LOG_FILE", str(log_path))
    monkeypatch.setattr(Config, "LOG_ACCESS", "1")
    monkeypatch.setattr(Config, "LOG_HEALTHCHECKS", "0")

    app = Flask("logging-request-test")

    @app.get("/hello")
    def hello() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/about")
    def about() -> dict[str, bool]:
        return {"ok": True}

    configure_logging(app)

    with app.test_client() as client:
        response = client.get(
            "/hello",
            headers={
                "X-Request-ID": "req-123",
                "X-Forwarded-For": "203.0.113.9, 127.0.0.1",
            },
        )
        assert response.headers["X-Request-ID"] == "req-123"
        client.get("/about")

    _flush_owned_handlers()
    records = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    request_records = [
        record
        for record in records
        if record["message"].startswith("[request]")
    ]

    assert len(request_records) == 1
    assert request_records[0]["request_id"] == "req-123"
    assert request_records[0]["method"] == "GET"
    assert request_records[0]["path"] == "/hello"
    assert request_records[0]["remote_addr"] == "203.0.113.9"
    assert request_records[0]["status_code"] == 200
    assert isinstance(request_records[0]["duration_ms"], float)


def _owned_root_handlers() -> list[logging.Handler]:
    return [
        handler
        for handler in logging.getLogger().handlers
        if getattr(handler, _OWNED_HANDLER_ATTR, False)
    ]


def _flush_owned_handlers() -> None:
    for handler in _owned_root_handlers():
        handler.flush()
