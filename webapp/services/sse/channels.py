"""
Progress channels for SSE streaming.

Provides thread-safe channels for publishing progress updates
from background tasks to SSE streams.
"""

from __future__ import annotations

import queue
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Optional, Tuple

from webapp.services.sse.messages import format_sse_message

# Type alias for queue items: (event_name, data)
_QueueItem = Tuple[str, Any] | None


@dataclass
class ProgressChannel:
    """
    A thread-safe channel for publishing progress updates.

    Use this to send updates from background processing to SSE streams.

    Example:
        channel = ProgressChannel()
        channel.send_progress(0, 100, "Starting...")
        channel.send_progress(50, 100, "Halfway...")
        channel.complete(result)
    """

    channel_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    _queue: queue.Queue[_QueueItem] = field(default_factory=queue.Queue)
    _closed: bool = field(default=False)

    def send(
        self,
        data: Any,
        event: str = "progress",
    ) -> None:
        """Send a progress update."""
        if not self._closed:
            self._queue.put((event, data))

    def send_progress(
        self,
        percent: int,
        message: Optional[str] = None,
    ) -> None:
        """
        Send a structured progress update.

        Args:
            percent: Progress percentage (0-100).
            message: Optional status message.
        """
        data: Dict[str, Any] = {"percent": percent}
        if message:
            data["message"] = message
        self.send(data, event="progress")

    def send_log(self, message: str, level: str = "info") -> None:
        """
        Send a log message.

        Args:
            message: The log message.
            level: Log level ('info', 'warning', 'error', 'debug').
        """
        self.send({"message": message, "level": level}, event="log")

    def send_error(self, error: str) -> None:
        """Send an error message."""
        self.send({"error": error}, event="error")

    def complete(
        self,
        data: Any = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Mark the channel as complete.

        Args:
            data: Optional result data to include in the complete event.
            error: Optional error message if processing failed.
        """
        if error:
            self.send({"error": error}, event="complete")
        else:
            self.send({"data": data}, event="complete")
        self._closed = True
        self._queue.put(None)  # Sentinel to stop iteration

    def close(self) -> None:
        """Close the channel without sending a complete event."""
        self._closed = True
        self._queue.put(None)

    @property
    def is_closed(self) -> bool:
        """Check if the channel is closed."""
        return self._closed

    def stream(self, timeout: float = 30.0) -> Generator[str, None, None]:
        """
        Generate SSE messages from the channel.

        This is a blocking generator that yields messages as they arrive.
        Use this in an SSE endpoint to stream updates to clients.

        Args:
            timeout: Timeout in seconds for waiting on messages.
                     Sends keepalive comments on timeout.

        Yields:
            Formatted SSE message strings.
        """
        while True:
            try:
                item = self._queue.get(timeout=timeout)
                if item is None:  # Sentinel
                    break
                event, data = item
                yield format_sse_message(data, event=event)
            except queue.Empty:
                if self._closed:
                    break
                # Send keepalive comment to prevent connection timeout
                yield ": keepalive\n\n"


class ChannelRegistry:
    """
    Registry for managing active SSE channels.

    This allows background tasks to publish to channels that SSE endpoints
    can stream from. Thread-safe for concurrent access.

    Example:
        registry = ChannelRegistry()
        channel = registry.create()
        # Pass channel.channel_id to client
        # Client connects to /stream/progress/{channel_id}
        # Background task uses channel.send_progress(...)
    """

    def __init__(self) -> None:
        self._channels: Dict[str, ProgressChannel] = {}
        self._lock = threading.Lock()

    def create(self) -> ProgressChannel:
        """Create and register a new channel."""
        channel = ProgressChannel()
        with self._lock:
            self._channels[channel.channel_id] = channel
        return channel

    def get(self, channel_id: str) -> Optional[ProgressChannel]:
        """Get a channel by ID."""
        with self._lock:
            return self._channels.get(channel_id)

    def remove(self, channel_id: str) -> None:
        """Remove a channel from the registry."""
        with self._lock:
            if channel_id in self._channels:
                self._channels[channel_id].close()
                del self._channels[channel_id]

    def cleanup_closed(self) -> int:
        """
        Remove all closed channels.

        Returns:
            Number of channels removed.
        """
        with self._lock:
            closed = [cid for cid, ch in self._channels.items() if ch.is_closed]
            for cid in closed:
                del self._channels[cid]
            return len(closed)

    def count(self) -> int:
        """Get the number of active channels."""
        with self._lock:
            return len(self._channels)


# Global registry instance
channels = ChannelRegistry()
