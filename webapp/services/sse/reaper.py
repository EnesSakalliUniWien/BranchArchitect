"""
Periodic sweep that evicts closed SSE channels from the registry.

A channel is only removed from ``ChannelRegistry`` by the
``/stream/progress/<channel_id>`` endpoint's request-scoped cleanup (see
``webapp.routes.routes.stream_progress``). If a client never opens that
endpoint after receiving a ``channel_id`` - a network drop, a page reload,
or the desktop app closing in the window between the POST response and the
EventSource connecting - the channel is still marked closed by its
background job (``ProgressChannel.complete``/``close``), but nothing ever
removes it from the registry. Over a long-running desktop session this
leaks a ``ProgressChannel`` (and its buffered message queue) per missed
connection.

This reaper sweeps already-*closed* channels on an interval so those
orphans get collected. It deliberately never touches channels that are
still active/unclosed, since a legitimate long-running tree-inference job
can stay open far longer than any reasonable sweep interval.
"""

from __future__ import annotations

import logging
import threading

from webapp.services.sse.channels import ChannelRegistry

log = logging.getLogger(__name__)

DEFAULT_SWEEP_INTERVAL_SECONDS = 60.0


class ChannelReaper:
    """Background thread that periodically evicts closed channels from a registry."""

    def __init__(
        self,
        registry: ChannelRegistry,
        interval_seconds: float = DEFAULT_SWEEP_INTERVAL_SECONDS,
    ) -> None:
        self._registry = registry
        self._interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the sweep thread. Safe to call more than once (no-op after the first)."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="sse-channel-reaper",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the sweep thread to stop on its next wake-up."""
        self._stop_event.set()

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_seconds):
            removed = self._registry.cleanup_closed()
            if removed:
                log.info("[sse-reaper] removed %d orphaned closed channel(s)", removed)
