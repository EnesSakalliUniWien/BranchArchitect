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
    if split in node.to_splits():
        return

    # Find the correct parent node where this split should be applied
    _apply_split_at_node(split, node)

    # Refresh split indices after modification
    root = node.get_root()
    root.initialize_split_indices(root.taxa_encoding)
    root.invalidate_caches()

    # Verify the split was applied
    tree_splits = root.to_splits()
    if split not in tree_splits:
        raise SplitApplicationError(
            split=split,
            tree_splits=list(tree_splits),
            message="Cannot apply split - incompatible with existing topology",
        )


def _apply_split_at_node(split: Partition, node: Node) -> bool:
    """
    Recursively find and apply split at the correct node.

    Returns True if split was applied at this node or a descendant.
    """
    split_indices = set(split.indices)

    # Check if split_indices is a proper subset of node.split_indices
    if split_indices < set(node.split_indices):
        remaining_children: list[Node] = []
        reassigned_children: list[Node] = []

        for child in node.children:
            child_split_set = set(child.split_indices)
            if child_split_set.issubset(split_indices):
                reassigned_children.append(child)
            else:
                remaining_children.append(child)

        # Create new node if we have multiple children to reassign
        if len(reassigned_children) > 1:
            new_node = Node(
                name="",
                split_indices=split,
                children=reassigned_children,
                length=0,
                taxa_encoding=node.taxa_encoding,
            )
            node.children = remaining_children
            # Use append_child to properly set parent pointer
            node.append_child(new_node)
            return True

    # Recursively try children
    for child in node.children:
        if child.children and _apply_split_at_node(split, child):
            return True

    return False


def _apply_split_no_rebuild(split: Partition, node: Node) -> bool:
    """
    Apply a split without rebuilding indices. Used for batch operations.

    Returns True if split was applied, False if it already exists or cannot be applied.

    Note: In rooted trees, a split and its complement are DIFFERENT nodes.
    We only check if the exact split exists, not its complement.
    The complement check was causing bugs where large splits (e.g., 50 out of 52 taxa)
    would fail to apply because their small complement (e.g., 2 taxa) already existed.
    """
    split_set = set(split.indices)
    split_mask = split.bitmask

    # Check if the EXACT split is already present - idempotent operation
    # Note: We do NOT check for complement here because in rooted trees,
    # a split and its complement represent different nodes in the tree.
    for n in node.traverse():
        if n.split_indices.bitmask == split_mask:
            # Split already exists - no action needed
            return False

    # Try applying direct split
    if _apply_split_at_node(split, node):
        return True

    # If direct failed, try applying complement split
    # This handles the case where we need to create a node for the "other side"
    # of the bipartition to achieve the desired topology.
    encoding = node.taxa_encoding
    all_indices = set(encoding.values())
    complement_indices = all_indices - split_set

    # Only try complement if it's a valid non-empty split
    if complement_indices:
        complement_split = split
        # Check if complement already exists before trying to apply
        complement_mask = complement_split.bitmask
        complement_exists = any(
            n.split_indices.bitmask == complement_mask for n in node.traverse()
        )
        if not complement_exists:
            if _apply_split_at_node(complement_split, node):
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
    3. Rebuilds indices ONCE at the end
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

    # Apply each split WITHOUT rebuilding indices (batch mode)
    applied_any = False
    for split in sorted_path:
        if _apply_split_no_rebuild(split, tree):
            applied_any = True

    # Rebuild indices ONCE after all splits are applied
    if applied_any:
        root = tree.get_root()
        root.initialize_split_indices(root.taxa_encoding)
        root.invalidate_caches()

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

    # Get base tree splits once for checking
    base_splits = base_tree.to_splits() if copy else grafted_tree.to_splits()

    # Apply splits in batch mode (no index rebuild per split)
    applied_any = False
    for ref_split in sorted_ref_path:
        if ref_split not in base_splits:
            if _apply_split_no_rebuild(ref_split, grafted_tree):
                applied_any = True
            else:
                logger.warning(
                    f"[Expand] Failed to apply split {list(ref_split.indices)} "
                    f"(Bitmask: {ref_split.bitmask:b}) to grafted tree. "
                    "This implies incompatibility with the current topology."
                )

    # Rebuild indices ONCE after all splits are applied
    if applied_any:
        root = grafted_tree.get_root()
        root.initialize_split_indices(root.taxa_encoding)
        root.invalidate_caches(propagate_up=True, propagate_down=True)

    return grafted_tree
