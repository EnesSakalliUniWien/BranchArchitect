"""
Logging configuration for the Flask application.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from flask import Flask, Response, g, has_request_context, request

from webapp.config import Config

_OWNED_HANDLER_ATTR = "_phylo_movies_handler"
_FALSE_VALUES = {"0", "false", "no", "off", "none", ""}
_TRUE_VALUES = {"1", "true", "yes", "on"}
_HEALTHCHECK_PATHS = {"/about", "/health"}


class RequestContextFilter(logging.Filter):
    """Attach request metadata to records when Flask request context exists."""

    def filter(self, record: logging.LogRecord) -> bool:
        if has_request_context():
            _set_record_default(record, "request_id", getattr(g, "request_id", "-"))
            _set_record_default(record, "method", request.method)
            _set_record_default(record, "path", request.path)
            _set_record_default(record, "remote_addr", _client_ip())
        else:
            _set_record_default(record, "request_id", "-")
            _set_record_default(record, "method", "-")
            _set_record_default(record, "path", "-")
            _set_record_default(record, "remote_addr", "-")

        _set_record_default(record, "status_code", None)
        _set_record_default(record, "duration_ms", None)
        return True


class JsonFormatter(logging.Formatter):
    """Small JSON formatter for container and log-aggregator friendly output."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
            "method": getattr(record, "method", "-"),
            "path": getattr(record, "path", "-"),
            "remote_addr": getattr(record, "remote_addr", "-"),
        }

        status_code = getattr(record, "status_code", None)
        if status_code is not None:
            payload["status_code"] = status_code

        duration_ms = getattr(record, "duration_ms", None)
        if duration_ms is not None:
            payload["duration_ms"] = duration_ms

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, separators=(",", ":"), default=str)


def configure_logging(app: Flask) -> None:
    """
    Configure backend logging for local, desktop, and container runtimes.

    Console logging is the primary sink because production containers and the
    Electron launcher already capture stdout/stderr. File logging is opt-in for
    normal Python runs and defaults to a user log file only for frozen bundles.
    """
    log_level, invalid_level = _resolve_log_level(Config.LOG_LEVEL)
    formatter = _build_formatter(Config.LOG_FORMAT)
    handlers = _build_handlers(log_level, formatter)

    root_logger = logging.getLogger()
    _remove_owned_handlers(root_logger)
    root_logger.setLevel(log_level)

    for handler in handlers:
        root_logger.addHandler(handler)

    # Let Flask app logs flow through the same root handler graph. This avoids
    # duplicate records and keeps app/module logging behavior consistent.
    _remove_owned_handlers(app.logger)
    app.logger.handlers = []
    app.logger.setLevel(logging.NOTSET)
    app.logger.propagate = True

    for module in ["brancharchitect", "msa_to_trees", "webapp", "waitress"]:
        module_logger = logging.getLogger(module)
        module_logger.setLevel(logging.NOTSET)

    logging.captureWarnings(True)
    _install_request_logging_hooks(app)

    if invalid_level is not None:
        app.logger.warning(
            "Invalid LOG_LEVEL=%r; falling back to INFO", invalid_level
        )

    sinks = ", ".join(_handler_sink(handler) for handler in handlers)
    app.logger.info(
        "Logging configured level=%s format=%s sinks=%s",
        logging.getLevelName(log_level),
        _normalized_log_format(Config.LOG_FORMAT),
        sinks,
    )


def _build_handlers(
    log_level: int, formatter: logging.Formatter
) -> list[logging.Handler]:
    handlers: list[logging.Handler] = []

    console_handler = logging.StreamHandler()
    _mark_owned(console_handler)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(RequestContextFilter())
    handlers.append(console_handler)

    log_file = _resolve_log_file()
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=5_000_000,
            backupCount=5,
            encoding="utf-8",
        )
        _mark_owned(file_handler)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(RequestContextFilter())
        handlers.append(file_handler)

    return handlers


def _build_formatter(configured_format: str) -> logging.Formatter:
    if _normalized_log_format(configured_format) == "json":
        return JsonFormatter()

    return logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )


def _resolve_log_level(value: str) -> tuple[int, str | None]:
    raw_value = str(value or "INFO").strip()
    if raw_value.isdecimal():
        return int(raw_value), None

    level_name = raw_value.upper()
    level = logging.getLevelName(level_name)
    if isinstance(level, int):
        return level, None
    return logging.INFO, raw_value


def _resolve_log_file() -> Path | None:
    configured = Config.LOG_FILE
    if configured is not None:
        configured = str(configured).strip()
        if configured.lower() in _FALSE_VALUES:
            return None
        return Path(configured).expanduser()

    if getattr(sys, "frozen", False):
        return Config.LOG_DIR / "backend.log"

    return None


def _install_request_logging_hooks(app: Flask) -> None:
    extensions = app.extensions.setdefault("phylo_movies_logging", {})
    if extensions.get("request_hooks_installed"):
        return

    @app.before_request
    def _start_request_log_context() -> None:
        g.request_started_at = time.perf_counter()
        g.request_id = _request_id()

    @app.after_request
    def _log_completed_request(response: Response) -> Response:
        request_id = getattr(g, "request_id", _request_id())
        response.headers["X-Request-ID"] = request_id

        if not _access_logging_enabled():
            return response
        if request.path in _HEALTHCHECK_PATHS and not _healthcheck_logging_enabled():
            return response

        started_at = getattr(g, "request_started_at", None)
        duration_ms = (
            round((time.perf_counter() - started_at) * 1000, 2)
            if started_at is not None
            else None
        )
        status_code = response.status_code
        level = _request_log_level(status_code)

        app.logger.log(
            level,
            "[request] %s %s -> %s in %sms",
            request.method,
            request.path,
            status_code,
            duration_ms if duration_ms is not None else "-",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.path,
                "remote_addr": _client_ip(),
                "status_code": status_code,
                "duration_ms": duration_ms,
            },
        )
        return response

    extensions["request_hooks_installed"] = True


def _request_log_level(status_code: int) -> int:
    if status_code >= 500:
        return logging.ERROR
    if status_code >= 400:
        return logging.WARNING
    return logging.INFO


def _request_id() -> str:
    header_value = request.headers.get("X-Request-ID", "").strip()
    if header_value:
        return header_value[:128]
    return uuid.uuid4().hex


def _client_ip() -> str:
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip()
    return request.headers.get("X-Real-IP") or request.remote_addr or "-"


def _access_logging_enabled() -> bool:
    return str(Config.LOG_ACCESS).strip().lower() not in _FALSE_VALUES


def _healthcheck_logging_enabled() -> bool:
    return str(Config.LOG_HEALTHCHECKS).strip().lower() in _TRUE_VALUES


def _normalized_log_format(value: str) -> str:
    configured = str(value or "text").strip().lower()
    if configured == "json":
        return "json"
    return "text"


def _set_record_default(record: logging.LogRecord, key: str, value: Any) -> None:
    if not hasattr(record, key):
        setattr(record, key, value)


def _mark_owned(handler: logging.Handler) -> None:
    setattr(handler, _OWNED_HANDLER_ATTR, True)


def _remove_owned_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if getattr(handler, _OWNED_HANDLER_ATTR, False):
            logger.removeHandler(handler)
            handler.close()


def _handler_sink(handler: logging.Handler) -> str:
    if isinstance(handler, RotatingFileHandler):
        return os.fspath(handler.baseFilename)
    return "stderr"
