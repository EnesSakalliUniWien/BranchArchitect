"""Validation checks for tree interpolation results."""

from __future__ import annotations

import logging

from brancharchitect.tree import Node


def assert_tree_topology_matches_destination(
    final_state: Node, destination_tree: Node, logger: logging.Logger
) -> None:
    """
    Verify that the final interpolation state matches destination topology.
    """
    final_splits = final_state.to_splits()
    destination_splits = destination_tree.to_splits()

    if final_splits == destination_splits:
        return

    missing = destination_splits - final_splits
    extra = final_splits - destination_splits
    msg = (
        "[tree_interpolation] Final topology mismatch: "
        f"expected {len(destination_splits)} splits, got {len(final_splits)}. "
        f"Missing splits: { {tuple(split.indices) for split in missing} } "
        f"Extra splits: { {tuple(split.indices) for split in extra} }"
    )
    logger.error(msg)
    raise ValueError(msg)
