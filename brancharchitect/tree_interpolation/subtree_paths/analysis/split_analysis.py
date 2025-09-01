"""Split analysis helpers for subtree-path interpolation."""

from typing import Any, Dict, Optional, Set, Tuple
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree import Node


def get_unique_splits_for_active_changing_edge_subtree(
    to_be_collapsed_tree: Node,
    expanded_tree: Node,
    active_changing_edge: Partition,
) -> Tuple[PartitionSet[Partition], PartitionSet[Partition]]:
    """
    Get splits within the active changing edge scope only.

    Returns (to_be_collapsed_splits, to_be_expanded_splits).
    """
    to_be_collapsed_node: Node | None = to_be_collapsed_tree.find_node_by_split(
        active_changing_edge
    )
    to_be_created_node: Node | None = expanded_tree.find_node_by_split(
        active_changing_edge
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
            f"Active changing edge {active_changing_edge} not found in either tree."
        )


def find_incompatible_splits(
    reference_splits: PartitionSet[Partition],
    candidate_splits: PartitionSet[Partition],
    max_size_ratio: Optional[float] = None,
) -> PartitionSet[Partition]:
    """Find all candidate splits incompatible with any reference split."""
    if not reference_splits or not candidate_splits:
        encoding = list(reference_splits)[0].encoding if reference_splits else {}
        return PartitionSet(encoding=encoding)

    encoding = list(reference_splits)[0].encoding
    incompatible_splits: PartitionSet[Partition] = PartitionSet(encoding=encoding)
    seen_bitmasks: Set[int] = set()

    for ref_split in reference_splits:
        ref_size = len(ref_split.indices) if max_size_ratio is not None else 0
        max_allowed_size = (
            int(ref_size * max_size_ratio)
            if max_size_ratio is not None
            else float("inf")
        )

        for candidate_split in candidate_splits:
            if candidate_split.bitmask in seen_bitmasks:
                continue
            if ref_split == candidate_split:
                continue
            if max_size_ratio is not None:
                candidate_size = len(candidate_split.indices)
                if candidate_size > max_allowed_size:
                    continue
            if not ref_split.is_compatible_with(candidate_split):
                incompatible_splits.add(candidate_split)
                seen_bitmasks.add(candidate_split.bitmask)

    return incompatible_splits


def get_shared_split_paths(
    path_split_paths: Dict[Partition, "PartitionSet[Partition]"],
) -> Dict[Partition, Dict[str, Any]]:
    """
    Build a shared-split_path index:
      { split_path -> {"split_path": split_path, "subtrees": set_of_subtrees} }
    Only keep split_paths that occur in at least two subtrees.
    """
    shared: Dict[Partition, Dict[str, Any]] = {}
    for subtree, split_paths in path_split_paths.items():
        for split_path in split_paths:
            info = shared.setdefault(
                split_path, {"split_path": split_path, "subtrees": set()}
            )
            info["subtrees"].add(subtree)

    return {
        split_path: info
        for split_path, info in shared.items()
        if len(info["subtrees"]) >= 2
    }

