"""
Sequential lattice interpolation public API.

Provides the stateful builder and wrapper that construct sequential
interpolations across adjacent tree pairs to create smooth animations
between phylogenetic trees.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Dict

from brancharchitect.elements.partition import Partition
from brancharchitect.tree import Node
from brancharchitect.core.tree_pairs import iter_consecutive_pairs
from brancharchitect.tree_interpolation.types import (
    TreeInterpolationSequence,
    TreePairInterpolation,
)
from brancharchitect.tree_interpolation.solution_mapping import (
    generate_solution_mappings,
)
from brancharchitect.tree_interpolation.pair_interpolation import (
    process_tree_pair_interpolation,
)

__all__: List[str] = [
    "SequentialInterpolationBuilder",
]


class SequentialInterpolationBuilder:
    """
    Stateful builder for constructing sequential lattice interpolations.

    Builds smooth phylogenetic tree animations by processing consecutive tree pairs
    and maintaining state for inspection and debugging. Features include:

    - Stateful design preserving intermediate results for analysis
    - Automatic state reset between builds to prevent cross-run contamination
    - Configurable pair processing with optional precomputed solutions
    - Comprehensive logging and performance tracking
    - Data integrity validation ensuring 1:1 correspondence between trees and metadata

    The builder processes tree pairs sequentially, generating interpolated trees
    between each consecutive pair and maintaining complete metadata tracking.
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        precomputed_pair_solutions: Optional[
            List[Optional[Dict[Partition, List[Partition]]]]
        ] = None,
    ):
        """
        Initialize the sequential interpolation builder.

        Args:
            logger: Logger instance for operation tracking.
                If None, uses the module's default logger.
            precomputed_pair_solutions: Pre-calculated lattice solutions for each pair.
                Can significantly speed up processing when available.
        """
        # Configure logging with fallback to module logger
        self.logger = logger or logging.getLogger(__name__)

        self.precomputed_pair_solutions = precomputed_pair_solutions
        self._initialize_build_state()

    def _initialize_build_state(self) -> None:
        """Initialize all state variables for a fresh interpolation build."""
        self.interpolated_trees: List[Node] = []
        self.source_mappings: List[Dict[Partition, Dict[Partition, Partition]]] = []
        self.target_mappings: List[Dict[Partition, Dict[Partition, Partition]]] = []
        self.current_pivot_edge_tracking: List[Optional[Partition]] = []
        self.pair_tree_counts: List[int] = []
        self.jumping_subtree_solutions: List[Dict[Partition, List[Partition]]] = []

    def _process_pair(
        self,
        t1: Node,
        t2: Node,
        pair_index: int,
        precomputed_solution: Optional[Dict[Partition, List[Partition]]],
    ) -> Node:
        """
        Process a single tree pair and collect the results into the builder's state.

        Returns:
            The final resolved tree from the interpolation, which should be used
            as the starting delimiter for the next pair to ensure sequence continuity.
        """
        pair_processing_start_time = time.perf_counter()
        interpolation_result: TreePairInterpolation = process_tree_pair_interpolation(
            t1.deep_copy(),
            t2.deep_copy(),
            precomputed_solution,
            pair_index=pair_index,
        )
        processing_duration = time.perf_counter() - pair_processing_start_time

        self.logger.debug(
            f"Processed T{pair_index}→T{pair_index + 1} in {processing_duration:.3f}s; generated {len(interpolation_result.trees)} trees"
        )

        # Collect results into stateful attributes
        self.interpolated_trees.extend(interpolation_result.trees)

        interpolated_tree_count = len(interpolation_result.trees)

        # Trees and tracking should have 1:1 correspondence from interpolation
        self.current_pivot_edge_tracking.extend(
            interpolation_result.current_pivot_edge_tracking
        )

        self.pair_tree_counts.append(interpolated_tree_count)

        self.jumping_subtree_solutions.append(
            interpolation_result.jumping_subtree_solutions or {}
        )

        # Build and store solution-to-atom mappings (destination/source) for this pair
        destination_map: Dict[Partition, Dict[Partition, Partition]] = {}
        source_map: Dict[Partition, Dict[Partition, Partition]] = {}
        if interpolation_result.jumping_subtree_solutions:
            destination_map, source_map = generate_solution_mappings(
                interpolation_result.jumping_subtree_solutions,
                target=t2,
                reference=t1,
            )
        self.source_mappings.append(destination_map)
        self.target_mappings.append(source_map)

        # Return the final resolved tree from this interpolation
        if len(interpolation_result.trees) > 0:
            return interpolation_result.trees[-1]

        # For identical trees (no interpolation), return the destination tree reordered
        # to match the current source ordering so that downstream delimiters stay aligned.
        self.logger.debug(
            f"No interpolation needed for pair {pair_index} - returning destination tree aligned to source order"
        )
        aligned_destination = t2.deep_copy()
        aligned_destination.reorder_taxa(list(t1.get_current_order()))
        return aligned_destination

    def _add_delimiter_frame(self, tree: Node) -> None:
        """Add an original tree and a None tracker to the sequence as a delimiter."""
        # Deep copy to create an independent snapshot
        self.interpolated_trees.append(tree.deep_copy())
        self.current_pivot_edge_tracking.append(None)

    def _finalize_sequence(self, original_tree_count: int) -> TreeInterpolationSequence:
        """Construct the final sequence object and log a summary."""
        self.logger.info(
            f"Completed interpolation sequence: {len(self.interpolated_trees)} total trees from {original_tree_count} originals + {sum(self.pair_tree_counts)} interpolated"
        )

        return TreeInterpolationSequence(
            interpolated_trees=self.interpolated_trees,
            mapping_one=self.source_mappings,
            mapping_two=self.target_mappings,
            current_pivot_edge_tracking=self.current_pivot_edge_tracking,
            pair_interpolated_tree_counts=self.pair_tree_counts,
            jumping_subtree_solutions_list=self.jumping_subtree_solutions,
        )

    def build(self, trees: List[Node]) -> TreeInterpolationSequence:
        """Build sequential interpolations between consecutive tree pairs."""
        if len(trees) < 2:
            raise ValueError("Need at least 2 trees for interpolation")

        # Initialize state for a fresh build
        self._initialize_build_state()

        self.logger.info(
            f"Building sequential lattice interpolations for {len(trees)} trees ({len(trees) - 1} pairs)"
        )

        # Add the first tree as the initial delimiter
        self._add_delimiter_frame(trees[0])

        for pair in iter_consecutive_pairs(trees):
            pair_index, source, target, is_first, is_last = pair

            precomputed_solution = (
                self.precomputed_pair_solutions[pair_index]
                if self.precomputed_pair_solutions is not None
                else None
            )
            source_tree = source if is_first else self.interpolated_trees[-1]

            # Process the pair and get the final resolved tree
            resolved_tree = self._process_pair(
                source_tree, target, pair_index, precomputed_solution
            )

            # Add the resolved tree as the delimiter for the next pair
            # This ensures: last frame of pair N = first frame of pair N+1
            if not is_last:
                self._add_delimiter_frame(resolved_tree)

        # Use the final resolved tree as the last delimiter to maintain sequence continuity
        self._add_delimiter_frame(self.interpolated_trees[-1])

        return self._finalize_sequence(len(trees))
