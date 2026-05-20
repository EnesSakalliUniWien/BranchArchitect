"""SSE transport contract for completed movie data."""

from __future__ import annotations

from logging import Logger
from typing import Any, Dict, List

from webapp.services.sse import ProgressChannel

MOVIE_METADATA_EVENT = "metadata"
DEFAULT_TREE_CHUNK_SIZE = 100


def send_movie_stream(
    channel: ProgressChannel,
    metadata: Dict[str, Any],
    trees: List[Dict[str, Any]],
    log: Logger,
    *,
    chunk_size: int = DEFAULT_TREE_CHUNK_SIZE,
) -> None:
    """Send one completed movie payload using the only live frontend contract."""
    total_trees = len(trees)
    log.info("[treedata/stream] Streaming %s trees in chunks", total_trees)
    channel.send_progress(90, "Streaming trees...")
    channel.send({"metadata": metadata}, event=MOVIE_METADATA_EVENT)
    channel.send_trees_chunked(trees, chunk_size=chunk_size)
    channel.send_progress(100, "Complete")
    channel.complete()
