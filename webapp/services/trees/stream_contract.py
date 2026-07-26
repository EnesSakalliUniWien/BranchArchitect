"""SSE transport contract for completed movie data."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Protocol

from webapp.services.sse import ProgressChannel

MOVIE_METADATA_EVENT = "metadata"
DEFAULT_TREE_CHUNK_SIZE = 100


class SupportsInfoLog(Protocol):
    def info(self, message: str, *args: object) -> None: ...


def send_movie_stream(
    channel: ProgressChannel,
    metadata: Dict[str, Any],
    trees: List[Dict[str, Any]],
    log: SupportsInfoLog,
    *,
    chunk_size: int = DEFAULT_TREE_CHUNK_SIZE,
) -> None:
    """Send one completed movie payload using the only live frontend contract."""
    total_trees = len(trees)
    chunk_count = (total_trees + chunk_size - 1) // chunk_size if chunk_size else 0
    log.info(
        "[treedata/stream] Streaming %s trees in %s chunk(s)",
        total_trees,
        chunk_count,
    )
    channel.send_progress(90, "Streaming trees...")
    channel.send({"metadata": metadata}, event=MOVIE_METADATA_EVENT)
    t_enqueue_start = time.perf_counter()
    channel.send_trees_chunked(trees, chunk_size=chunk_size)
    log.info(
        "[PhaseTimer] send_trees_chunked_enqueue %.3fs",
        time.perf_counter() - t_enqueue_start,
    )
    channel.send_progress(100, "Complete")
    channel.complete()
    log.info("[treedata/stream] Completed movie stream")
