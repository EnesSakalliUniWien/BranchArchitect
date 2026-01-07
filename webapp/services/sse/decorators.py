"""
Decorators for SSE-enabled background tasks.

Provides utilities for running functions in background threads
with automatic progress channel management.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Concatenate, ParamSpec, TypeVar

from webapp.services.sse.channels import ProgressChannel, channels

P = ParamSpec("P")
T = TypeVar("T")


def with_progress(
    func: Callable[Concatenate[ProgressChannel, P], T],
) -> Callable[P, ProgressChannel]:
    """
    Decorator to run a function in a background thread with progress channel.

    The decorated function receives a ProgressChannel as its first argument.
    The function runs in a daemon thread, and exceptions are automatically
    sent to the channel as error events.

    Args:
        func: Function to decorate. Must accept ProgressChannel as first arg.

    Returns:
        Wrapper function that returns a ProgressChannel immediately.

    Example:
        @with_progress
        def long_task(channel: ProgressChannel, data: dict):
            channel.send_progress(0, 100, "Starting...")
            # ... do work ...
            channel.send_progress(50, 100, "Halfway...")
            # ... more work ...
            channel.complete({"status": "done"})

        # In route:
        @app.route('/process', methods=['POST'])
        def process():
            channel = long_task(request.json)
            return jsonify({"channel_id": channel.channel_id})

        # Client connects to /stream/progress/{channel_id} for updates
    """

    def wrapper(*args: Any, **kwargs: Any) -> ProgressChannel:
        channel = channels.create()

        def run() -> None:
            try:
                func(channel, *args, **kwargs)
            except Exception as e:
                channel.send_error(str(e))
                channel.close()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

        return channel

    # Preserve function metadata
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__

    return wrapper
