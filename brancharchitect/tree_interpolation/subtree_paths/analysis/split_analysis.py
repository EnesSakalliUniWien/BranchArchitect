"""Split analysis helpers for subtree-path interpolation."""

import logging
from typing import Optional, Set, Tuple
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree import Node

logger = logging.getLogger(__name__)


def get_unique_splits_for_current_pivot_edge_subtree(
    source_tree: Node,
    destination_tree: Node,
    current_pivot_edge: Partition,
) -> Tuple[PartitionSet[Partition], PartitionSet[Partition]]:
    """
    Get splits within the active changing edge scope only.

    Returns (to_be_collapsed_splits, to_be_expanded_splits).
    """
    to_be_collapsed_node: Node | None = source_tree.find_node_by_split(
        current_pivot_edge
    )
    to_be_created_node: Node | None = destination_tree.find_node_by_split(
        current_pivot_edge
    )

    if to_be_collapsed_node and to_be_created_node:
        original_collapse_splits: PartitionSet[Partition] = (
            to_be_collapsed_node.to_splits()
        )
        original_expand_splits: PartitionSet[Partition] = to_be_created_node.to_splits()

        to_be_collapsed_splits = original_collapse_splits - original_expand_splits
        to_be_expanded_splits = original_expand_splits - original_collapse_splits

        return to_be_collapsed_splits, to_be_expanded_splits
    else:
        raise ValueError(
            f"Current pivot edge {current_pivot_edge} not found in either tree."
        )


def find_incompatible_splits(
    destination_splits: PartitionSet[Partition],
    source_splits: PartitionSet[Partition],
    max_size_ratio: Optional[float] = None,
) -> PartitionSet[Partition]:
    """Find all source splits incompatible with any destination split.

    Raises:
        ValueError: If encodings differ between destination and source sets.
    """
    if not destination_splits or not source_splits:
        # If either set is empty, no incompatibilities exist
        if destination_splits:
            encoding = destination_splits.encoding
        elif source_splits:
            encoding = source_splits.encoding
        else:
            encoding = {}
        return PartitionSet(encoding=encoding)

    encoding = destination_splits.encoding
    all_indices = set(encoding.values())
    incompatible_splits: PartitionSet[Partition] = PartitionSet(encoding=encoding)
    seen_bitmasks: Set[int] = set()

    for dst_split in destination_splits:
        dst_size = len(dst_split.indices) if max_size_ratio is not None else 0
        max_allowed_size = (
            int(dst_size * max_size_ratio)
            if max_size_ratio is not None
            else float("inf")
        )

        for src_split in source_splits:
            if src_split.bitmask in seen_bitmasks:
                continue
            if dst_split == src_split:
                continue
            if max_size_ratio is not None:
                src_size = len(src_split.indices)
                if src_size > max_allowed_size:
                    continue
            if not dst_split.is_compatible_with(src_split, all_indices):
                incompatible_splits.add(src_split)
                seen_bitmasks.add(src_split.bitmask)
                logger.debug(
                    f"[SplitAnalysis] Incompatible found: Src {src_split.indices} vs Dst {dst_split.indices}"
                )

    return incompatible_splits
