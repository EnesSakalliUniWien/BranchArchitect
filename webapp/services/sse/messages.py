"""
SSE message formatting utilities.

Provides functions to format data as Server-Sent Event messages
and create Flask responses for SSE streaming.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator, Optional

from flask import Response

from brancharchitect.io import UUIDEncoder


def format_sse_message(
    data: Any,
    event: Optional[str] = None,
    event_id: Optional[str] = None,
    retry: Optional[int] = None,
) -> str:
    """
    Format data as an SSE message.

    Args:
        data: The data to send (will be JSON-encoded if not a string).
        event: Optional event type (e.g., 'progress', 'error', 'complete').
        event_id: Optional event ID for client reconnection.
        retry: Optional retry interval in milliseconds.

    Returns:
        Formatted SSE message string.

    Example:
        >>> format_sse_message({'count': 1}, event='progress')
        'event: progress\\ndata: {"count": 1}\\n\\n'
    """
    lines = []

    if event_id is not None:
        lines.append(f"id: {event_id}")

    if event is not None:
        lines.append(f"event: {event}")

    if retry is not None:
        lines.append(f"retry: {retry}")

    # Serialize data
    if isinstance(data, str):
        payload = data
    else:
        payload = json.dumps(data, cls=UUIDEncoder)

    # SSE requires each line of data to be prefixed with "data: "
    for line in payload.split("\n"):
        lines.append(f"data: {line}")

    # Messages are terminated by double newline
    return "\n".join(lines) + "\n\n"


def sse_response(
    generator: Iterator[str],
    headers: Optional[Dict[str, str]] = None,
) -> Response:
    """
    Create a Flask Response for SSE streaming.

    Args:
        generator: An iterator yielding SSE-formatted messages.
        headers: Optional additional headers.

    Returns:
        Flask Response configured for SSE.

    Example:
        def generate():
            for i in range(10):
                yield format_sse_message({'count': i})
        return sse_response(generate())
    """
    # Note: "Connection" is a hop-by-hop header forbidden by PEP 3333 (WSGI)
    # Waitress/WSGI handles connection management automatically
    default_headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "X-Accel-Buffering": "no",  # Disable nginx buffering
    }

    if headers:
        default_headers.update(headers)

    return Response(
        generator,
        mimetype="text/event-stream",
        headers=default_headers,
    )
