"""
Data types and classes for tree interpolation.

This module contains the data structures used throughout the tree interpolation
process, including result containers and intermediate data representations.
"""

from __future__ import annotations
from dataclasses import dataclass
from brancharchitect.elements.partition import Partition
from brancharchitect.tree import Node
from typing import List, Dict, Optional
from .pair_data import PairData


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

    Tree Naming Convention:
    - Original trees: "T0", "T1", "T2", ...
    - S-edge interpolation: "IT{i}_down_{j}", "C{i}_{j}", "C{i}_{j}_reorder",
      "IT{i}_up_{j}", "IT{i}_ref_{j}" where i is tree index, j is s-edge number
    - Classical fallback: "IT{i}_classical_{j}_1" through "IT{i}_classical_{j}_5"

    S-edge Tracking:
    - Original trees: None (no s-edge applied)
    - Interpolated trees: Specific Partition representing the s-edge being processed
    - Classical interpolation: None (doesn't use specific s-edges)

    Attributes:
        interpolated_trees: Complete sequence of all trees (originals + interpolated)
        interpolation_sequence_labels: Human-readable names for each tree in sequence
        mapping_one: Target-to-atom solution mappings for each tree pair
        mapping_two: Reference-to-atom solution mappings for each tree pair
        s_edge_tracking: S-edge applied for each tree (None for originals/classical)
        s_edge_lengths: Number of interpolation steps generated per tree pair
        lattice_solutions_list: Raw jumping taxa algorithm results per pair
        s_edge_distances_list: Distance metrics from components to s-edges per pair

    Example:
        # For 3 input trees where T0≠T1 (2 s-edges found), T1=T2 (0 s-edges found):
        # Tree sequence: T0, [10 interpolated], T1, T2
        # Total trees: 3 + 10 + 0 = 13 trees (NOT 28!)

        result = build_sequential_lattice_interpolations([tree1, tree2, tree3])
        print(f"Total trees: {result.total_interpolated_trees}")  # 13 (conditional!)
        print(f"Pairs processed: {result.get_pair_count()}")      # 2
        print(f"S-edge lengths: {result.s_edge_lengths}")         # [10, 0] (second pair identical)
        print(f"Tree at index 11: {result.interpolation_sequence_labels[11]}")  # "T1"
    """

    # Core interpolation results
    interpolated_trees: List[Node]
    interpolation_sequence_labels: List[str]

    # Per-pair mapping data
    mapping_one: List[Dict[Partition, Partition]]
    mapping_two: List[Dict[Partition, Partition]]

    # S-edge tracking and metadata
    s_edge_tracking: List[Optional[Partition]]
    subtree_tracking: List[Optional[Partition]]
    s_edge_lengths: List[int]

    # Algorithm results and distance metrics
    lattice_solutions_list: List[Dict[Partition, List[List[Partition]]]]
    s_edge_distances_list: List[Dict[Partition, Dict[str, float]]]

    def get_pair_count(self) -> int:
        """
        Get the number of consecutive tree pairs processed during interpolation.

        For N input trees, this returns N-1 pairs (T0->T1, T1->T2, ..., T(N-2)->T(N-1)).

        Returns:
            Number of tree pairs that were interpolated between
        """
        return len(self.s_edge_lengths)

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
            - s_edge_distances: Distance metrics from components to s-edges

        Raises:
            IndexError: If pair_index is out of valid range

        Example:
            pair_data = result.get_pair_data(0)  # Data for T0->T1 interpolation
            print(f"This pair has {pair_data['s_edge_length']} interpolation steps")
        """
        if pair_index >= self.get_pair_count():
            raise IndexError(
                f"Pair index {pair_index} out of range (0-{self.get_pair_count() - 1})"
            )

        return {
            "mapping_one": self.mapping_one[pair_index],
            "mapping_two": self.mapping_two[pair_index],
            "s_edge_length": self.s_edge_lengths[pair_index],
            "lattice_solutions": self.lattice_solutions_list[pair_index],
            "s_edge_distances": self.s_edge_distances_list[pair_index],
            "subtree_tracking": [self.subtree_tracking[pair_index]],
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
            Sum of s_edge_lengths across all processed tree pairs
        """
        return sum(self.s_edge_lengths)

    def get_original_tree_indices(self) -> List[int]:
        """
        Get global indices of original (non-interpolated) trees in the sequence.

        Returns:
            List of indices where s_edge_tracking[i] is None, indicating original trees
        """
        return [i for i, s_edge in enumerate(self.s_edge_tracking) if s_edge is None]

    def get_interpolated_tree_indices(self) -> List[int]:
        """
        Get global indices of interpolated trees in the sequence.

        Returns:
            List of indices where s_edge_tracking[i] is not None, indicating interpolated trees
        """
        return [
            i for i, s_edge in enumerate(self.s_edge_tracking) if s_edge is not None
        ]

    def get_classical_interpolation_indices(self) -> List[int]:
        """
        Get global indices of trees generated by classical interpolation fallback.

        Classical interpolation is used when s-edge processing fails, not when
        trees are identical (identical trees generate zero interpolation trees).

        Returns:
            List of indices where tree names contain "_classical_"
        """
        return [
            i
            for i, name in enumerate(self.interpolation_sequence_labels)
            if "_classical_" in name
        ]

    def get_zero_interpolation_pairs(self) -> List[int]:
        """
        Get pair indices that generated zero interpolation trees (identical trees).

        Returns:
            List of pair indices where s_edge_lengths[i] == 0
        """
        return [i for i, length in enumerate(self.s_edge_lengths) if length == 0]
