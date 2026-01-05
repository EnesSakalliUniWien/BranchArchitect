"""
Logging configuration for the Flask application.
"""

import logging
from logging.handlers import RotatingFileHandler

from flask import Flask

from webapp.config import Config


def configure_logging(app: Flask) -> None:
    """
    Attach both file and console handlers to the Flask app logger.

    Handlers:
        - File handler: Rotating log file (1 MiB, 3 backups) at logs/backend.log
        - Console handler: Human-readable output for local development

    Args:
        app: The Flask application instance.
    """
    # Ensure the log directory exists
    Config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    # File handler (rotating)
    backend_file = Config.LOG_DIR / "backend.log"
    backend_handler = RotatingFileHandler(
        backend_file,
        maxBytes=1_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    backend_handler.setLevel(logging.DEBUG)
    backend_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    backend_handler.setFormatter(backend_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        "%H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers on reload
    if not any(isinstance(h, RotatingFileHandler) for h in root_logger.handlers):
        root_logger.handlers = []
        root_logger.addHandler(backend_handler)
        root_logger.addHandler(console_handler)

    # Configure Flask's app.logger
    if not any(isinstance(h, RotatingFileHandler) for h in app.logger.handlers):
        app.logger.handlers = []
        app.logger.addHandler(backend_handler)
        app.logger.addHandler(console_handler)

    app.logger.setLevel(logging.DEBUG)
    app.logger.propagate = False

    # Configure module loggers
    for module in ["brancharchitect", "webapp"]:
        module_logger = logging.getLogger(module)
        module_logger.setLevel(logging.DEBUG)

    app.logger.info(f"Logging configured. Log file: {backend_file}")
