"""Configuration for the Flask application."""

import os
import secrets
import sys
import platform
from pathlib import Path


def get_log_dir() -> Path:
    """Get the appropriate log directory based on environment."""
    if getattr(sys, "frozen", False):
        # Running in a bundle - use user's log directory
        home = Path.home()
        system = platform.system()
        
        if system == "Darwin":
            return home / "Library/Logs/PhyloMovies"
        elif system == "Windows":
            return home / "AppData/Local/PhyloMovies/Logs"
        else:
            # Linux/Unix
            return home / ".local/share/phylomovies/logs"
    
    # Development mode - use local logs directory
    return Path("logs")


def _default_cors_origins() -> list[str]:
    """
    Default allowed CORS origins.

    The server binds to 127.0.0.1 only, but a wildcard origin still lets
    any webpage open in the user's browser make cross-origin requests to
    it while the app is running. Instead we allow only the origins this
    app actually loads from:
      - "null": the Origin header a browser sends for requests from a
        file:// page, which is how the packaged Electron app loads its
        production build (see RFC 6454 - opaque origins serialize to the
        literal string "null", not the word None).
      - The Vite dev server and preview server ports used during local
        development.
    """
    return [
        "null",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ]


def _parse_cors_origins(raw: str | None) -> list[str] | str:
    """Parse the CORS_ORIGINS env var, falling back to the safe default list."""
    if raw is None or raw == "":
        return _default_cors_origins()
    if raw == "*":
        return "*"
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


class Config:
    """Flask configuration."""

    # Flask settings
    # For Electron apps, generate a random secret key if not provided
    SECRET_KEY = os.environ.get("SECRET_KEY") or secrets.token_hex(32)

    DEBUG = os.environ.get("FLASK_DEBUG", "1") == "1"

    # CORS settings - allowlisted for the app's own origins by default.
    # Set CORS_ORIGINS="*" explicitly (env var) to opt back into a wildcard,
    # or a comma-separated list to customize the allowlist.
    CORS_ORIGINS = _parse_cors_origins(os.environ.get("CORS_ORIGINS"))

    # File upload settings
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size

    # Logging
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    LOG_FORMAT = os.environ.get("LOG_FORMAT", "text")
    LOG_ACCESS = os.environ.get("LOG_ACCESS", "1")
    LOG_HEALTHCHECKS = os.environ.get("LOG_HEALTHCHECKS", "0")
    LOG_DIR = get_log_dir()
    LOG_FILE = os.environ.get("BACKEND_LOG_FILE") or os.environ.get("LOG_FILE")
