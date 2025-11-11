"""
Data types and classes for tree interpolation.

This module contains the data structures used throughout the tree interpolation
process, including result containers and intermediate data representations.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from brancharchitect.elements.partition import Partition
from brancharchitect.tree import Node

from .pair_data import PairData

MappingDict = dict[Partition, dict[Partition, Partition]]
JumpingSolutions = dict[Partition, list[Partition]]


def _empty_node_list() -> list[Node]:
    return []


def _empty_mapping_list() -> list[MappingDict]:
    return []


def _empty_partition_list() -> list[Optional[Partition]]:
    return []


def _empty_int_list() -> list[int]:
    return []


def _empty_jumping_solutions() -> list[JumpingSolutions]:
    return []


@dataclass
class TreeInterpolationSequence:
    """
    Comprehensive result structure from sequential lattice-based tree interpolation.

    This dataclass encapsulates all data generated during the interpolation of a sequence
    of phylogenetic trees. It replaces complex tuple returns with a clear, structured
    format that groups related data logically and provides convenient access methods.

    Core Structure:
    - For N input trees, generates N + sum(s_edges_per_pair * 5) interpolated trees
    - Each tree pair (Ti, Ti+1) produces 0 to many interpolation trees depending on s-edges found
    - If Ti and Ti+1 are identical: 0 s-edges found → 0 interpolation trees generated
    - If Ti and Ti+1 differ: k s-edges found → k*5 interpolation trees generated
    - Classical interpolation fallback produces exactly 5 trees when s-edge processing fails

    Active Changing Split Tracking:
    - Original trees: None (no active changing split applied)
    - Interpolated trees: Specific Partition representing the active changing split being processed
    - Classical interpolation: None (doesn't use specific active changing splits)

    Attributes:
        interpolated_trees: Complete sequence of all trees (originals + interpolated)
        mapping_one: Target-to-atom solution mappings for each tree pair
            (outer key = pivot edge, inner key = solution partition)
        mapping_two: Reference-to-atom solution mappings for each tree pair
            (outer key = pivot edge, inner key = solution partition)
        active_changing_split_tracking: S-edge applied for each tree (None for originals/classical)
        pair_interpolated_tree_counts: Total interpolated trees generated per pair
        jumping_subtree_solutions_list: Raw jumping taxa algorithm results per pair
        # distances removed

    Example:
        # For 3 input trees where T0≠T1 (2 s-edges found), T1=T2 (0 s-edges found):
        # Tree sequence: T0, [10 interpolated], T1, T2
        # Total trees: 3 + 10 + 0 = 13 trees (NOT 28!)

        result = build_sequential_lattice_interpolations([tree1, tree2, tree3])
        # result.total_interpolated_trees -> 13 (conditional!)
        # result.pair_interpolated_tree_counts -> [10, 0]
        # result.get_pair_count() -> 2
    """

    # Core interpolation results
    interpolated_trees: list[Node] = field(default_factory=_empty_node_list)
    mapping_one: list[MappingDict] = field(default_factory=_empty_mapping_list)
    mapping_two: list[MappingDict] = field(default_factory=_empty_mapping_list)
    current_pivot_edge_tracking: list[Optional[Partition]] = field(
        default_factory=_empty_partition_list
    )
    pair_interpolated_tree_counts: list[int] = field(default_factory=_empty_int_list)
    jumping_subtree_solutions_list: list[JumpingSolutions] = field(
        default_factory=_empty_jumping_solutions
    )

    def get_pair_count(self) -> int:
        """
        Get the number of consecutive tree pairs processed during interpolation.

        For N input trees, this returns N-1 pairs (T0->T1, T1->T2, ..., T(N-2)->T(N-1)).

        Returns:
            Number of tree pairs that were interpolated between
        """
        return len(self.pair_interpolated_tree_counts)


    def get_pair_data(self, pair_index: int) -> PairData:
        """
        Get comprehensive interpolation data for a specific tree pair.

        Retrieves all interpolation-related data for the tree pair at the given index,
        including lattice solutions, atom mappings, s-edge sequences, and distance metrics.

        Args:
            pair_index: Zero-based index of the tree pair (0 for T0->T1, 1 for T1->T2, etc.)

        Returns:
            Dictionary containing:
            - mapping_one: Target tree solution-to-atom mappings
            - mapping_two: Reference tree solution-to-atom mappings
            - s_edge_length: Number of interpolation steps for this pair
            - lattice_solutions: Raw jumping taxa algorithm results
            # distances removed

        Raises:
            IndexError: If pair_index is out of valid range

        Example:
            pair_data = result.get_pair_data(0)  # Data for T0->T1 interpolation
            # pair_data['s_edge_length'] -> number of steps for the pair
        """
        if pair_index >= self.get_pair_count():
            raise IndexError(
                f"Pair index {pair_index} out of range (0-{self.get_pair_count() - 1})"
            )

        return {
            "mapping_one": self.mapping_one[pair_index],
            "mapping_two": self.mapping_two[pair_index],
            "s_edge_length": self.pivot_edge_lengths[pair_index],
            "jumping_subtree_solutions": self.jumping_subtree_solutions_list[
                pair_index
            ],
        }

    @property
    def total_interpolated_trees(self) -> int:
        """
        Total number of trees in the complete interpolation sequence.

        Includes both original trees and all interpolated trees generated
        from the lattice-based s-edge processing and classical fallbacks.

        Returns:
            Total count of trees in interpolated_trees list
        """
        return len(self.interpolated_trees)

    @property
    def total_interpolation_steps(self) -> int:
        """
        Total number of interpolation steps across all tree pairs.

        Each s-edge generates exactly 5 interpolation steps, and classical
        interpolation fallbacks also generate exactly 5 steps. This property
        sums up all interpolation work done across all pairs.

        Returns:
            Sum of total interpolated trees across all processed pairs
        """
        return sum(self.pair_interpolated_tree_counts)

    def get_original_tree_indices(self) -> list[int]:
        """
        Get global indices of original (non-interpolated) trees in the sequence.

        Returns:
            List of indices where active_changing_split_tracking[i] is None, indicating original trees
        """
        return [
            i
            for i, s_edge in enumerate(self.current_pivot_edge_tracking)
            if s_edge is None
        ]

    def get_interpolated_tree_indices(self) -> list[int]:
        """
        Get global indices of interpolated trees in the sequence.

        Returns:
            List of indices where active_changing_split_tracking[i] is not None, indicating interpolated trees
        """
        return [
            i
            for i, s_edge in enumerate(self.current_pivot_edge_tracking)
            if s_edge is not None
        ]

    @property
    def current_pivt_edge_trackeing(self) -> list[Optional[Partition]]:
        """Backward-compatible alias for legacy typo."""
        return self.current_pivot_edge_tracking

    @current_pivt_edge_trackeing.setter
    def current_pivt_edge_trackeing(
        self, value: list[Optional[Partition]]
    ) -> None:
        self.current_pivot_edge_tracking = value

    def get_classical_interpolation_indices(self) -> list[int]:
        """
        Get global indices of trees generated by classical interpolation fallback.

        Classical interpolation is used when s-edge processing fails, not when
        trees are identical (identical trees generate zero interpolation trees).

        Returns:
            Empty list (labels removed). Add alternative marker if needed later.
        """
        return []

    def get_zero_interpolation_pairs(self) -> list[int]:
        """
        Get pair indices that generated zero interpolation trees (identical trees).

        Returns:
            List of pair indices where pair_interpolated_tree_counts[i] == 0
        """
        return [
            i
            for i, length in enumerate(self.pair_interpolated_tree_counts)
            if length == 0
        ]

    @property
    def pivot_edge_lengths(self) -> list[int]:
        """Number of interpolation steps (per pair)."""
        return self.pair_interpolated_tree_counts
