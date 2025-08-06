"""Configuration for the Flask application."""

import os
from pathlib import Path


class Config:
    """Flask configuration."""
    
    # Flask settings
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
    DEBUG = os.environ.get("FLASK_DEBUG", "1") == "1"
    
    # CORS settings
    CORS_ORIGINS = "*"
    
    # File upload settings
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size
    
    # Logging
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    LOG_DIR = Path("logs")
    LOG_FILE = LOG_DIR / "debug_log.html"