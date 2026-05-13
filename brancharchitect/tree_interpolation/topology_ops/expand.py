"""
Expand Operations - Adding Splits to Trees

This module handles topology expansion by applying new splits to trees.
Use this when you need to add internal nodes (increase tree resolution).

Public API:
    - SplitApplicationError: Exception raised when split application fails
    - apply_split_simple: Apply a single split to a tree (no retry, fail-fast)
    - execute_expand_path: Apply multiple splits in size order (largest first)
    - create_subtree_grafted_tree: Create a new tree with additional splits grafted on

Related modules:
    - collapse.py: For removing splits (reducing topology)
    - weights.py: For manipulating branch lengths
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import List
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.split_semantics import same_rooted_split
from brancharchitect.tree import Node

__all__ = [
    "SplitApplicationError",
    "apply_split_simple",
    "execute_expand_path",
    "create_subtree_grafted_tree",
]

logger = logging.getLogger(__name__)


@dataclass
class SplitApplicationError(Exception):
    """Error raised when split application fails.

    Provides diagnostic information including the split that failed,
    the current tree splits, and a descriptive message.
    """

    split: Partition
    tree_splits: List[Partition] = field(default_factory=list)
    message: str = "Split application failed"

    def __str__(self) -> str:
        taxa_names = [self.split.reverse_encoding[i] for i in self.split.indices]
        lines = [
            self.message,
            f"Split: {list(self.split.indices)} = ({', '.join(taxa_names)})",
            f"Tree has {len(self.tree_splits)} splits",
        ]
        if self.tree_splits:
            lines.append("Current tree splits (first 5):")
            for s in self.tree_splits[:5]:
                s_names = [s.reverse_encoding[i] for i in s.indices]
                lines.append(f"  - {list(s.indices)} = ({', '.join(s_names)})")
        return "\n".join(lines)


def apply_split_simple(split: Partition, node: Node) -> None:
    """
    Apply a split to a tree without retry or validation.

    This is a simplified version that trusts the planning phase has computed
    compatible paths. If the split cannot be applied, it raises an error
    immediately rather than attempting automatic conflict resolution.

    Args:
        split: The partition to apply to the tree
        node: The root node of the tree

    Raises:
        SplitApplicationError: If split cannot be applied (incompatible topology)

    Note:
        Encoding is guaranteed consistent by the interpolation pipeline -
        both trees are parsed with the same encoding at the start.
    """
    # Check if split is already present - idempotent operation
    split_index = _get_tree_split_index(node)
    if split in split_index:
        return

    # Find the correct parent node where this split should be applied
    _apply_split_at_node(split, node, split_index)

    # Verify the split was applied
    root = node.get_root()
    tree_splits = root.to_splits()
    if split not in tree_splits:
        raise SplitApplicationError(
            split=split,
            tree_splits=list(tree_splits),
            message="Cannot apply split - incompatible with existing topology",
        )


def _get_tree_split_index(node: Node) -> dict[Partition, Node]:
    """Return this tree's cached split index, building it only when needed."""
    root = node.get_root()
    if root._split_index is None:
        root.build_split_index()
    if root._split_index is None:
        return {}
    return root._split_index


def _is_subset_mask(child_mask: int, parent_mask: int) -> bool:
    return (child_mask & ~parent_mask) == 0


def _find_split_application_parent(
    split: Partition, split_index: dict[Partition, Node]
) -> Node | None:
    """
    Find the narrowest existing node that can receive `split`.

    A valid parent strictly contains the new split and has at least two direct
    children whose clades fit inside the new split. Those children can then be
    grouped under the new internal node.
    """
    split_mask = split.bitmask
    best_parent: Node | None = None
    best_parent_size = float("inf")

    for candidate in split_index.values():
        candidate_mask = candidate.split_indices.bitmask
        if candidate_mask == split_mask:
            continue
        if not _is_subset_mask(split_mask, candidate_mask):
            continue

        reassigned_child_count = 0
        for child in candidate.children:
            if _is_subset_mask(child.split_indices.bitmask, split_mask):
                reassigned_child_count += 1
                if reassigned_child_count > 1:
                    break

        if reassigned_child_count <= 1:
            continue

        candidate_size = candidate.split_indices.size
        if candidate_size < best_parent_size:
            best_parent = candidate
            best_parent_size = candidate_size

    return best_parent


def _apply_split_to_parent(split: Partition, parent: Node) -> Node:
    split_mask = split.bitmask
    remaining_children: list[Node] = []
    reassigned_children: list[Node] = []

    for child in parent.children:
        if _is_subset_mask(child.split_indices.bitmask, split_mask):
            reassigned_children.append(child)
        else:
            remaining_children.append(child)

    new_node = Node(
        name="",
        split_indices=split,
        children=reassigned_children,
        length=0,
        taxa_encoding=parent.taxa_encoding,
    )
    parent.children = remaining_children
    new_node.parent = parent
    parent.children.append(new_node)
    parent.invalidate_caches(propagate_up=True, propagate_down=False)
    return new_node


def _apply_split_at_node(
    split: Partition, node: Node, split_index: dict[Partition, Node] | None = None
) -> bool:
    """
    Find and apply split at the correct node.

    Returns True if split was applied at this node or a descendant.
    """
    if split_index is None:
        split_index = _get_tree_split_index(node)

    parent = _find_split_application_parent(split, split_index)
    if parent is None:
        return False

    new_node = _apply_split_to_parent(split, parent)
    parent.get_root()._split_index = split_index
    split_index[new_node.split_indices] = new_node
    return True


def _apply_split_no_rebuild(
    split: Partition, node: Node, split_index: dict[Partition, Node] | None = None
) -> bool:
    """
    Apply a split without rebuilding indices. Used for batch operations.

    Returns True if split was applied, False if it already exists or cannot be applied.

    Existing-split checks use exact rooted splits only. If the requested split
    cannot be applied directly, the operation fails without trying the complement
    because rooted interpolation treats a split and its complement as different
    tree nodes.
    """
    # Check if the EXACT split is already present - idempotent operation
    # Note: We do NOT check for complement here because in rooted trees,
    # a split and its complement represent different nodes in the tree.
    if split_index is None:
        split_index = _get_tree_split_index(node)
    if split in split_index and same_rooted_split(
        split_index[split].split_indices, split
    ):
        # Split already exists - no action needed
        return False

    # Try applying direct split
    if _apply_split_at_node(split, node, split_index):
        return True

    return False


def execute_expand_path(
    tree: Node,
    expand_path: List[Partition],
    reference_weights: dict[Partition, float] | None = None,
) -> Node:
    """
    Execute expand path by applying splits in size order.

    This function:
    1. Sorts splits by size (largest first)
    2. Applies each split sequentially (batch mode - no index rebuild per split)
    3. Updates the split lookup as each split is inserted
    4. Applies reference weights to new nodes
    5. Fails fast on any error

    Args:
        tree: The tree to modify (will be mutated)
        expand_path: Splits to apply
        reference_weights: Weights to apply to new nodes

    Returns:
        The modified tree with expand splits added

    Raises:
        SplitApplicationError: If any split fails to apply

    Requirements: 3.1, 3.2, 3.3, 3.4
    """
    if not expand_path:
        return tree

    # Sort by partition size (largest first), tie-break by bitmask for determinism
    sorted_path = sorted(expand_path, key=lambda p: (-len(p.indices), p.bitmask))
    split_index = _get_tree_split_index(tree)

    # Apply each split WITHOUT rebuilding indices (batch mode)
    for split in sorted_path:
        _apply_split_no_rebuild(split, tree, split_index)

    # Verify all splits were applied
    tree_splits = tree.to_splits()

    for split in sorted_path:
        if split not in tree_splits:
            raise SplitApplicationError(
                split=split,
                tree_splits=list(tree_splits),
                message="Cannot apply split - incompatible with existing topology",
            )

    # Apply reference weights if provided
    if reference_weights:
        for split in expand_path:
            node = tree.find_node_by_split(split)
            if node is not None:
                node.length = reference_weights.get(split, 0.0)

    return tree


def create_subtree_grafted_tree(
    base_tree: Node,
    ref_path_to_build: list[Partition],
    copy: bool = True,  # whether to copy the tree first
) -> Node:
    """
    Create grafted tree with order-preserving split application.

    Args:
        base_tree: Tree to graft onto
        ref_path_to_build: Splits to apply
        copy: If True, copy the tree first. If False, modify in place.
    """
    # Sort by partition size (number of taxa) in descending order
    # This ensures larger splits are applied before smaller ones
    sorted_ref_path = sorted(
        ref_path_to_build, key=lambda p: len(p.indices), reverse=True
    )

    grafted_tree = base_tree.deep_copy() if copy else base_tree
    split_index = _get_tree_split_index(grafted_tree)

    # Apply splits in batch mode (no index rebuild per split)
    for ref_split in sorted_ref_path:
        if ref_split not in split_index:
            if not _apply_split_no_rebuild(ref_split, grafted_tree, split_index):
                logger.warning(
                    f"[Expand] Failed to apply split {list(ref_split.indices)} "
                    f"(Bitmask: {ref_split.bitmask:b}) to grafted tree. "
                    "This implies incompatibility with the current topology."
                )

    return grafted_tree
