"""
Partial ordering strategy for subtree interpolation.

This module provides a modular approach to reordering trees during interpolation,
focusing on local subtree contexts rather than global tree ordering.
"""

from __future__ import annotations
import logging
from typing import List, Optional, Tuple
from brancharchitect.tree import Node, ReorderStrategy
from brancharchitect.elements.partition import Partition

logger = logging.getLogger(__name__)


class PartialOrderingStrategy:
    """
    Manages partial ordering of subtrees during interpolation.

    This strategy ensures that only the relevant local context is reordered
    during subtree interpolation, preserving the global tree structure.
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def determine_reorder_scope(
        self,
        active_changing_edge: Partition,
        create_path: List[Partition],
        collapse_path: List[Partition],
    ) -> Tuple[Partition, str]:
        """Determine the optimal scope for reordering."""
        if create_path:
            return create_path[-1], "last_expansion"
        if collapse_path:
            return collapse_path[0], "first_collapse"
        return active_changing_edge, "active_edge"

    def extract_local_taxa_order(
        self,
        node: Node,
        target_split: Partition,
        reference_tree: Optional[Node] = None,
    ) -> List[str]:
        """Extract the taxa order for a specific local context."""
        target_node = node.find_node_by_split(target_split)
        if not target_node:
            self.logger.warning(f"Target split {target_split} not found in tree")
            return list(node.get_current_order())

        local_order = list(target_node.get_current_order())
        if reference_tree:
            ref_node = reference_tree.find_node_by_split(target_split)
            if ref_node:
                ref_order = list(ref_node.get_current_order())
                common_taxa = set(local_order) & set(ref_order)
                local_order = [t for t in ref_order if t in common_taxa]
        return local_order

    def compute_ordering_distance(
        self,
        order1: List[str],
        order2: List[str],
    ) -> float:
        """Compute a distance measure between two orders (Kendall tau style)."""
        common = set(order1) & set(order2)
        if len(common) < 2:
            return 0.0
        pos1 = {taxon: i for i, taxon in enumerate(order1) if taxon in common}
        pos2 = {taxon: i for i, taxon in enumerate(order2) if taxon in common}
        inversions = 0
        taxa_list = sorted(common)
        for i in range(len(taxa_list)):
            for j in range(i + 1, len(taxa_list)):
                t1, t2 = taxa_list[i], taxa_list[j]
                if (pos1[t1] < pos1[t2]) != (pos2[t1] < pos2[t2]):
                    inversions += 1
        max_inversions = len(taxa_list) * (len(taxa_list) - 1) / 2
        return inversions / max_inversions if max_inversions > 0 else 0.0

    def generate_partial_order(
        self,
        tree: Node,
        reference_tree: Node,
        target_split: Partition,
        scope: str = "local",
    ) -> List[str]:
        """Generate a partial ordering for a specific split."""
        if scope == "global":
            return list(reference_tree.get_current_order())
        tree_node = tree.find_node_by_split(target_split)
        ref_node = reference_tree.find_node_by_split(target_split)
        if not tree_node:
            self.logger.warning(f"Target split {target_split} not found in tree")
            return list(tree.get_current_order())
        if not ref_node:
            self.logger.warning(f"Target split {target_split} not found in reference")
            return list(tree_node.get_current_order())
        tree_taxa = set(tree_node.get_current_order())
        ref_taxa = set(ref_node.get_current_order())
        common_taxa = tree_taxa & ref_taxa
        ref_order = list(ref_node.get_current_order())
        partial_order = [t for t in ref_order if t in common_taxa]
        remaining = tree_taxa - ref_taxa
        if remaining:
            t_order = list(tree_node.get_current_order())
            partial_order.extend([t for t in t_order if t in remaining])
        return partial_order


class AdaptiveReorderingStrategy(PartialOrderingStrategy):
    """Adaptive reordering strategy that uses a distance threshold."""

    def __init__(self, distance_threshold: float = 0.3):
        super().__init__()
        self.distance_threshold = distance_threshold

    def should_reorder(self, current_order: List[str], target_order: List[str]) -> bool:
        return self.compute_ordering_distance(current_order, target_order) > self.distance_threshold


def create_reordering_strategy(strategy: str = "adaptive", **kwargs) -> PartialOrderingStrategy:
    if strategy == "adaptive":
        return AdaptiveReorderingStrategy(**kwargs)
    return PartialOrderingStrategy()


def apply_partial_reordering(
    tree: Node,
    reference_tree: Node,
    active_changing_edge: Partition,
    create_path: List[Partition],
    collapse_path: List[Partition],
    strategy: Optional[PartialOrderingStrategy] = None,
) -> Node:
    """
    Apply partial reordering based on the interpolation context.

    Focuses on relevant local regions around the active-changing split to minimize
    disruption to the overall tree structure.
    """
    strategy = strategy or PartialOrderingStrategy()
    target_split, _ = strategy.determine_reorder_scope(
        active_changing_edge, create_path, collapse_path
    )
    # Compute a local target order and merge into a full permutation
    partial_order = strategy.generate_partial_order(
        tree,
        reference_tree,
        target_split,
        scope="local",
    )

    current_global = list(tree.get_current_order())
    po_set = set(partial_order)
    # Merge: place taxa in partial_order first, then the remaining taxa in their current order
    full_permutation = list(partial_order) + [t for t in current_global if t not in po_set]

    new_tree = tree.deep_copy()
    new_tree.reorder_taxa(full_permutation, strategy=ReorderStrategy.MINIMUM)
    return new_tree
