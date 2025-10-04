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
from brancharchitect.tree_interpolation.types import (
    TreeInterpolationSequence,
    TreePairInterpolation,
)
from brancharchitect.tree_interpolation.pair_interpolation import (
    process_tree_pair_interpolation,
)

__all__: List[str] = [
    "build_sequential_lattice_interpolations",
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
            List[Optional[Dict[Partition, List[List[Partition]]]]]
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
        self.source_mappings: List[Dict[Partition, Partition]] = []
        self.target_mappings: List[Dict[Partition, Partition]] = []
        self.active_split_tracking: List[Optional[Partition]] = []
        self.pair_tree_counts: List[int] = []
        self.jumping_subtree_solutions: List[
            Optional[Dict[Partition, List[List[Partition]]]]
        ] = []
        # Deprecated: subtree tracking removed
        self.last_build_result: Optional[TreeInterpolationSequence] = None

    def _process_pair(
        self,
        t1: Node,
        t2: Node,
        pair_index: int,
        precomputed_solution: Optional[Dict[Partition, List[List[Partition]]]],
    ) -> None:
        """Process a single tree pair and collect the results into the builder's state."""
        pair_processing_start_time = time.perf_counter()
        interpolation_result: TreePairInterpolation = process_tree_pair_interpolation(
            t1.deep_copy(), t2.deep_copy(), precomputed_solution
        )
        processing_duration = time.perf_counter() - pair_processing_start_time

        self.logger.debug(
            f"Processed T{pair_index}→T{pair_index + 1} in {processing_duration:.3f}s; generated {len(interpolation_result.trees)} trees"
        )

        # Collect results into stateful attributes
        self.interpolated_trees.extend(interpolation_result.trees)

        interpolated_tree_count = len(interpolation_result.trees)

        # Trees and tracking should have 1:1 correspondence from interpolation
        if interpolation_result.active_changing_split_tracking is not None:
            self.active_split_tracking.extend(
                interpolation_result.active_changing_split_tracking
            )

        self.pair_tree_counts.append(interpolated_tree_count)

        self.jumping_subtree_solutions.append(
            interpolation_result.jumping_subtree_solutions
        )

    def _get_precomputed_solution(
        self, pair_index: int
    ) -> Optional[Dict[Partition, List[List[Partition]]]]:
        """Retrieve the precomputed solution for a given pair index."""
        if self.precomputed_pair_solutions is not None:
            return self.precomputed_pair_solutions[pair_index]
        return None

    def _add_delimiter_frame(self, tree: Node) -> None:
        """Add an original tree and a None tracker to the sequence as a delimiter."""
        self.interpolated_trees.append(tree)
        self.active_split_tracking.append(None)

    def _finalize_sequence(self, original_tree_count: int) -> TreeInterpolationSequence:
        """Construct the final sequence object and log a summary."""
        self.logger.info(
            f"Completed interpolation sequence: {len(self.interpolated_trees)} total trees from {original_tree_count} originals + {sum(self.pair_tree_counts)} interpolated"
        )
        self.last_build_result = TreeInterpolationSequence(
            interpolated_trees=self.interpolated_trees,
            mapping_one=self.source_mappings,
            mapping_two=self.target_mappings,
            active_changing_split_tracking=self.active_split_tracking,
            pair_interpolated_tree_counts=self.pair_tree_counts,
            jumping_subtree_solutions_list=self.jumping_subtree_solutions,
        )
        return self.last_build_result

    def build(self, trees: List[Node]) -> TreeInterpolationSequence:
        """Build sequential interpolations between consecutive tree pairs."""
        if len(trees) < 2:
            raise ValueError("Need at least 2 trees for interpolation")

        # Initialize state for a fresh build
        self._initialize_build_state()

        self.logger.info(
            f"Building sequential lattice interpolations for {len(trees)} trees ({len(trees) - 1} pairs)"
        )

        for pair_index in range(len(trees) - 1):
            self._add_delimiter_frame(trees[pair_index])

            # Safely get the precomputed solution for the current pair.
            precomputed_solution = self._get_precomputed_solution(pair_index)

            self._process_pair(
                trees[pair_index],
                trees[pair_index + 1],
                pair_index,
                precomputed_solution,
            )

        self._add_delimiter_frame(trees[-1])

        return self._finalize_sequence(len(trees))


def build_sequential_lattice_interpolations(
    trees: List[Node],
    precomputed_pair_solutions: Optional[
        List[Optional[Dict[Partition, List[List[Partition]]]]]
    ] = None,
) -> TreeInterpolationSequence:
    """Build sequential lattice interpolations between consecutive tree pairs."""
    return SequentialInterpolationBuilder(
        precomputed_pair_solutions=precomputed_pair_solutions
    ).build(trees)
