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


class Config:
    """Flask configuration."""

    # Flask settings
    # For Electron apps, generate a random secret key if not provided
    SECRET_KEY = os.environ.get("SECRET_KEY") or secrets.token_hex(32)

    DEBUG = os.environ.get("FLASK_DEBUG", "1") == "1"

    # CORS settings - permissive for local Electron app
    CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")

    # File upload settings
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size

    # Logging
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    LOG_FORMAT = os.environ.get("LOG_FORMAT", "text")
    LOG_ACCESS = os.environ.get("LOG_ACCESS", "1")
    LOG_HEALTHCHECKS = os.environ.get("LOG_HEALTHCHECKS", "0")
    LOG_DIR = get_log_dir()
    LOG_FILE = os.environ.get("BACKEND_LOG_FILE") or os.environ.get("LOG_FILE")
