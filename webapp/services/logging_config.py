# --------------------------------------------------------------
#  logging_config.py
# --------------------------------------------------------------
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask
from ..config import Config


def configure_logging(app: Flask) -> None:
    """Attach both *file* and *console* handlers to ``app.logger``.

    *   **File handler** - HTML-wrapped `debug_log.html` (rotates at 1 MiB,
        keeps 5 backups).
    *   **Console handler** - colour-less, human-readable output for local dev.
    """

    # Ensure the log directory exists *before* touching the file.
    Config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    # -------- Single file handler (rotating) - plaintext backend.log in logs directory --------
    # Ensure logs directory exists and use consistent location
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

    # -------- Console handler (optional) --------
    # Keep console at INFO; start script redirects stdout to the same backend.log
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        "%H:%M:%S",
    )
    console_handler.setFormatter(console_formatter)

    # Configure the root logger so ALL modules use the same configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Clear existing handlers to avoid duplicates on reload
    if not any(isinstance(h, RotatingFileHandler) for h in root_logger.handlers):
        root_logger.handlers = []
        root_logger.addHandler(backend_handler)
        root_logger.addHandler(console_handler)

    # Also configure Flask's app.logger
    if not any(isinstance(h, RotatingFileHandler) for h in app.logger.handlers):
        app.logger.handlers = []
        app.logger.addHandler(backend_handler)
        app.logger.addHandler(console_handler)

    app.logger.setLevel(logging.DEBUG)
    app.logger.propagate = False

    # Configure specific module loggers to ensure they use DEBUG level
    for module in ["brancharchitect", "webapp"]:
        module_logger = logging.getLogger(module)
        module_logger.setLevel(logging.DEBUG)

    app.logger.info(f"Logging configured. Log file: {backend_file}")
