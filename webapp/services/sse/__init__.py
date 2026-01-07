"""
Server-Sent Events (SSE) support for Flask 3.x.

This package provides native SSE streaming without external dependencies.

Usage:
    from webapp.services.sse import sse_response, format_sse_message, channels

    @app.route('/stream')
    def stream():
        def generate():
            yield format_sse_message({'status': 'started'})
            yield format_sse_message({'progress': 50})
            yield format_sse_message({'status': 'complete'})
        return sse_response(generate())
"""

from webapp.services.sse.messages import format_sse_message, sse_response
from webapp.services.sse.channels import ProgressChannel, ChannelRegistry, channels
from webapp.services.sse.decorators import with_progress

__all__ = [
    "format_sse_message",
    "sse_response",
    "ProgressChannel",
    "ChannelRegistry",
    "channels",
    "with_progress",
]
