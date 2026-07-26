from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from brancharchitect.elements.partition import Partition
from brancharchitect.tree import Node

PendingFrame = tuple[Node, Optional[Partition], list[Partition]]


@dataclass(slots=True)
class FrameBatch:
    """Owns the frame lists produced by one interpolation step."""

    trees: list[Node] = field(default_factory=list)
    edges: list[Optional[Partition]] = field(default_factory=list)
    subtree_highlights: list[list[Partition]] = field(default_factory=list)

    def append(
        self,
        tree: Node,
        edge: Optional[Partition],
        highlight_group: list[Partition],
    ) -> None:
        self.trees.append(tree)
        self.edges.append(edge)
        self.subtree_highlights.append(list(highlight_group))


class PendingFrameBuffer:
    """
    Buffers a frame until the next phase decides whether it needs replacing.

    This keeps snapshot ownership rules in one place: a frame is either pending
    and still replaceable, or flushed into FrameBatch and immutable to callers.
    """

    def __init__(self, batch: FrameBatch):
        self.batch = batch
        self._pending: PendingFrame | None = None

    @property
    def has_pending(self) -> bool:
        return self._pending is not None

    def is_pending_tree(self, tree: Node) -> bool:
        return self._pending is not None and self._pending[0] is tree

    def flush(self) -> None:
        if self._pending is None:
            return

        tree, edge, highlight_group = self._pending
        self.batch.append(tree, edge, highlight_group)
        self._pending = None

    def set(
        self,
        tree: Node,
        edge: Optional[Partition],
        highlight_group: list[Partition],
    ) -> None:
        self.flush()
        self._pending = (tree, edge, list(highlight_group))

    def replace(
        self,
        tree: Node,
        edge: Optional[Partition],
        highlight_group: list[Partition],
    ) -> None:
        self._pending = (tree, edge, list(highlight_group))

    def snapshot_if_pending_tree_is(self, tree: Node) -> None:
        if self._pending is None or self._pending[0] is not tree:
            return

        _tree, edge, highlight_group = self._pending
        self._pending = (
            tree.deep_copy(build_split_index=False),
            edge,
            list(highlight_group),
        )
