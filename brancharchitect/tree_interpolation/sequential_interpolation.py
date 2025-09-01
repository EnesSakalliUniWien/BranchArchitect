"""
Sequential lattice interpolation public API.

Provides the stateful builder and wrapper that construct sequential
interpolations across adjacent tree pairs to create smooth animations
between phylogenetic trees.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Dict

from brancharchitect.elements.partition import Partition
from brancharchitect.tree import Node
from brancharchitect.tree_interpolation.types import (
    TreeInterpolationSequence,
)
from brancharchitect.tree_interpolation.active_changing_split_interpolation import (
    build_active_changing_split_interpolation_sequence,
)

logger: logging.Logger = logging.getLogger(__name__)

__all__: List[str] = [
    "build_sequential_lattice_interpolations",
    "SequentialInterpolationBuilder",
]


# collect_pair_results helper is no longer necessary with the stateful builder.


class SequentialInterpolationBuilder:
    """
    Stateful builder that constructs sequential interpolations across adjacent tree pairs.

    - Keeps intermediate lists as instance attributes for inspection after build().
    - Resets internal state at the start of each build() to avoid cross-run leakage.
    - Uses the module's pair processor by default, but can be customized.
    """

    def __init__(self, pair_processor=None, logger: Optional[logging.Logger] = None):
        # Default to the active-changing split interpolation function
        self.pair_processor = (
            pair_processor or build_active_changing_split_interpolation_sequence
        )
        self.logger = logger or logging.getLogger(__name__)
        self._reset()

    def _reset(self) -> None:
        self.results: List[Node] = []
        self.names: List[str] = []
        self.mappings_one: List[Dict[Partition, Partition]] = []
        self.mappings_two: List[Dict[Partition, Partition]] = []
        self.s_edge_tracking: List[Optional[Partition]] = []
        self.s_edge_lengths: List[int] = []
        self.lattice_solutions_list: List[Dict[Partition, List[List[Partition]]]] = []
        self.s_edge_distances_list: List[Dict[Partition, Dict[str, float]]] = []
        self.subtree_tracking: List[Optional[Partition]] = []
        self.last_result: Optional[TreeInterpolationSequence] = None

    def build(self, tree_list: List[Node]) -> TreeInterpolationSequence:
        if len(tree_list) < 2:
            raise ValueError("Need at least 2 trees for interpolation")

        # Reset state for a fresh build
        self._reset()

        self.logger.info(
            f"Building sequential lattice interpolations for {len(tree_list)} trees ({len(tree_list) - 1} pairs)"
        )

        for i in range(len(tree_list) - 1):
            # Add original tree to sequence
            self.results.append(tree_list[i])
            self.names.append(f"T{i}")
            self.s_edge_tracking.append(None)
            self.subtree_tracking.append(None)

            # Process tree pair (work on copies to avoid side effects)
            pair_result = self.pair_processor(
                tree_list[i].deep_copy(), tree_list[i + 1].deep_copy(), i
            )

            self.logger.info(
                f"Processed T{i}→T{i + 1}: generated {len(pair_result.trees)} interpolation trees"
            )

            # Collect results into stateful attributes
            self.results.extend(pair_result.trees)
            self.names.extend(pair_result.names)
            self.mappings_one.append(pair_result.mapping_one)
            self.mappings_two.append(pair_result.mapping_two)

            if pair_result.s_edge_tracking is not None:
                self.s_edge_tracking.extend(pair_result.s_edge_tracking)
                self.s_edge_lengths.append(len(pair_result.s_edge_tracking))
            else:
                self.s_edge_lengths.append(0)

            if pair_result.subtree_tracking is not None:
                self.subtree_tracking.extend(pair_result.subtree_tracking)

            if pair_result.lattice_edge_solutions is not None:
                self.lattice_solutions_list.append(pair_result.lattice_edge_solutions)
            if pair_result.s_edge_distances is not None:
                self.s_edge_distances_list.append(pair_result.s_edge_distances)

        # Add the final original tree to complete the sequence
        final_tree_index: int = len(tree_list) - 1
        self.results.append(tree_list[-1])
        self.names.append(f"T{final_tree_index}")
        self.s_edge_tracking.append(None)
        self.subtree_tracking.append(None)
        self.logger.info(
            f"Completed interpolation sequence: {len(self.results)} total trees from {len(tree_list)} originals + {sum(self.s_edge_lengths)} interpolated"
        )

        self.last_result = TreeInterpolationSequence(
            interpolated_trees=self.results,
            interpolation_sequence_labels=self.names,
            mapping_one=self.mappings_one,
            mapping_two=self.mappings_two,
            s_edge_tracking=self.s_edge_tracking,
            s_edge_lengths=self.s_edge_lengths,
            lattice_solutions_list=self.lattice_solutions_list,
            s_edge_distances_list=self.s_edge_distances_list,
            subtree_tracking=self.subtree_tracking,
        )
        return self.last_result


def build_sequential_lattice_interpolations(
    tree_list: List[Node],
) -> TreeInterpolationSequence:
    """Backwards-compatible wrapper that uses the stateful builder."""
    return SequentialInterpolationBuilder().build(tree_list)
