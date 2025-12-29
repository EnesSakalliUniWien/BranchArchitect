"""
Data types and classes for tree interpolation.

This module contains the data structures used throughout the tree interpolation
process, including result containers and intermediate data representations.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from itertools import groupby
from typing import Optional, Dict, List, Sequence, Tuple

from brancharchitect.elements.partition import Partition
from brancharchitect.tree import Node
from .pair_key import PairKey
from .tree_pair_solution import TreePairSolution, SplitChangeEvent
from .tree_meta_data import TreeMetadata

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


def _empty_pair_ranges() -> list[list[int]]:
    return []


def _empty_pair_solutions() -> dict[str, TreePairSolution]:
    return {}


def _empty_tree_metadata() -> list[TreeMetadata]:
    return []


@dataclass
class TreeInterpolationSequence:
    """
    Comprehensive result structure from sequential lattice-based tree interpolation.

    This dataclass encapsulates all data generated during the interpolation of a sequence
    of phylogenetic trees. It replaces complex tuple returns with a clear, structured
    format that groups related data logically and provides convenient access methods.

    Core Structure:
    - For N input trees, generates N + sum(pivot_edges_per_pair * 5) interpolated trees
    - Each tree pair (Ti, Ti+1) produces 0 to many interpolation trees depending on pivot edges found
    - If Ti and Ti+1 are identical: 0 pivot edges found → 0 interpolation trees generated
    - If Ti and Ti+1 differ: k pivot edges found → k*5 interpolation trees generated
    - Classical interpolation fallback produces exactly 5 trees when pivot edge processing fails

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

        from brancharchitect.tree_interpolation.sequential_interpolation import SequentialInterpolationBuilder
        result = SequentialInterpolationBuilder().build([tree1, tree2, tree3])
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
    # Tracks which subtree is being moved for each tree in the sequence
    # Parallel to current_pivot_edge_tracking: None for original trees, Partition for interpolated
    current_subtree_tracking: list[Optional[Partition]] = field(
        default_factory=_empty_partition_list
    )
    pair_interpolated_tree_counts: list[int] = field(default_factory=_empty_int_list)
    jumping_subtree_solutions_list: list[JumpingSolutions] = field(
        default_factory=_empty_jumping_solutions
    )
    tree_pair_solutions: Dict[str, TreePairSolution] = field(
        default_factory=_empty_pair_solutions
    )
    tree_metadata: list[TreeMetadata] = field(default_factory=_empty_tree_metadata)
    pair_interpolation_ranges: list[list[int]] = field(
        default_factory=_empty_pair_ranges
    )

    def get_pair_count(self) -> int:
        """
        Get the number of consecutive tree pairs processed during interpolation.

        For N input trees, this returns N-1 pairs (T0->T1, T1->T2, ..., T(N-2)->T(N-1)).

        Returns:
            Number of tree pairs that were interpolated between
        """
        return len(self.pair_interpolated_tree_counts)

    def get_pair_ranges(self, original_tree_indices: list[int]) -> list[list[int]]:
        """Compute global index ranges [start, end] for each pair's interpolated trees."""
        pair_count = len(self.jumping_subtree_solutions_list)
        if len(original_tree_indices) < pair_count + 1:
            raise IndexError(
                "Not enough original tree delimiters to key solutions "
                f"(have {len(original_tree_indices)}, need {pair_count + 1})"
            )
        bounded = original_tree_indices[: pair_count + 1]
        return [[bounded[i], bounded[i + 1]] for i in range(pair_count)]

    def build_pair_solutions(
        self,
        original_tree_indices: list[int],
        logger: Optional[logging.Logger] = None,
    ) -> Tuple[Dict[str, TreePairSolution], List[List[int]]]:
        """Build keyed TreePairSolution dict and pair ranges from the sequence data."""
        pair_ranges = self.get_pair_ranges(original_tree_indices)
        tree_pair_solutions: Dict[str, TreePairSolution] = {}

        for pair_index, (start, end) in enumerate(pair_ranges):
            pair_key = str(PairKey.from_index(pair_index))
            source_global_idx = start
            target_global_idx = end
            pivot_sequence = self.current_pivot_edge_tracking[
                source_global_idx + 1 : target_global_idx
            ]
            # Filter out None to keep ancestor list contiguous and events aligned
            ancestor_sequence: List[Partition] = [
                p for p in pivot_sequence if p is not None
            ]
            split_change_events = self._build_split_change_events(
                ancestor_sequence, source_global_idx, target_global_idx
            )

            pair_solution: TreePairSolution = {
                "jumping_subtree_solutions": self.jumping_subtree_solutions_list[
                    pair_index
                ],
                "solution_to_destination_map": self.mapping_one[pair_index],
                "solution_to_source_map": self.mapping_two[pair_index],
                "split_change_events": split_change_events,
                "source_tree_global_index": source_global_idx,
                "target_tree_global_index": target_global_idx,
                "interpolation_start_global_index": source_global_idx + 1,
            }
            tree_pair_solutions[pair_key] = pair_solution

            if logger:
                logger.debug(
                    "[pair_solutions] pair=%s range=[%d,%d] splits=%d",
                    pair_key,
                    start,
                    end,
                    len(split_change_events),
                )

        return tree_pair_solutions, pair_ranges

    @staticmethod
    def _build_split_change_events(
        split_sequence: Sequence[Optional[Partition]],
        source_global_idx: int,
        target_global_idx: int,
    ) -> List[SplitChangeEvent]:
        """Aggregate contiguous occurrences of a split into SplitChangeEvent entries.

        Retains None entries as gaps; they are skipped in event aggregation.
        """
        if not split_sequence:
            return []

        events: List[SplitChangeEvent] = []
        start_idx = 0

        for split, group in groupby(split_sequence):
            group_size = sum(1 for _ in group)
            if split is None:
                start_idx += group_size
                continue
            events.append(
                {
                    "split": split,
                    "step_range": (start_idx, start_idx + group_size - 1),
                    "source_tree_global_index": source_global_idx,
                    "target_tree_global_index": target_global_idx,
                }
            )
            start_idx += group_size

        return events

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
            for i, pivot_edge in enumerate(self.current_pivot_edge_tracking)
            if pivot_edge is None
        ]

    def get_interpolated_tree_indices(self) -> list[int]:
        """
        Get global indices of interpolated trees in the sequence.

        Returns:
            List of indices where active_changing_split_tracking[i] is not None, indicating interpolated trees
        """
        return [
            i
            for i, pivot_edge in enumerate(self.current_pivot_edge_tracking)
            if pivot_edge is not None
        ]

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
