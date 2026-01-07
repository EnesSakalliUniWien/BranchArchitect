"""
Webapp services package.

This package provides modular services for the Flask application:
- logging: Application logging configuration
- msa: MSA parsing and window parameter inference
- serialization: JSON serialization utilities
- sse: Server-Sent Events for real-time streaming
- trees: Tree processing and frontend data building
"""

# Re-export commonly used items for convenience
from webapp.services.logging import configure_logging
from webapp.services.trees import handle_tree_content, MovieData
from webapp.services.sse import (
    format_sse_message,
    sse_response,
    channels,
    ProgressChannel,
    with_progress,
)

__all__ = [
    # Logging
    "configure_logging",
    # Trees
    "handle_uploaded_file",
    "MovieData",
    # SSE
    "format_sse_message",
    "sse_response",
    "channels",
    "ProgressChannel",
    "with_progress",
]
