"""
Sequential lattice interpolation public API.

Provides the stateful builder and wrapper that construct sequential
interpolations across adjacent tree pairs to create smooth animations
between phylogenetic trees.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Dict, Callable, Any

from brancharchitect.elements.partition import Partition
from brancharchitect.tree import Node
from brancharchitect.tree_interpolation.types import (
    TreeInterpolationSequence,
)
from brancharchitect.tree_interpolation.active_changing_split_interpolation import (
    build_active_changing_split_interpolation_sequence,
)

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

    def __init__(
        self,
        pair_processor: Optional[
            Callable[
                [Node, Node, int, Optional[Dict[Partition, List[List[Partition]]]]], Any
            ]
        ] = None,
        logger: Optional[logging.Logger] = None,
        precomputed_pair_solutions: Optional[
            List[Optional[Dict[Partition, List[List[Partition]]]]]
        ] = None,
    ):
        # Default to the active-changing split interpolation function
        self.pair_processor: Callable[
            [Node, Node, int, Optional[Dict[Partition, List[List[Partition]]]]], Any
        ] = pair_processor or build_active_changing_split_interpolation_sequence
        self.logger = logger or logging.getLogger(__name__)
        self.precomputed_pair_solutions = precomputed_pair_solutions
        self._reset()

    def _reset(self) -> None:
        self.results: List[Node] = []
        self.mappings_one: List[Dict[Partition, Partition]] = []
        self.mappings_two: List[Dict[Partition, Partition]] = []
        self.active_changing_split_tracking: List[Optional[Partition]] = []
        self.pair_interpolated_tree_counts: List[int] = []
        self.lattice_solutions_list: List[Dict[Partition, List[List[Partition]]]] = []
        # Deprecated: subtree tracking removed
        self.last_result: Optional[TreeInterpolationSequence] = None

    @staticmethod
    def _normalize_len(
        seq: Optional[List[Optional[Partition]]],
        target_len: int,
        fill: Optional[Partition] = None,
        label: str = "seq",
        logger: Optional[logging.Logger] = None,
    ) -> List[Optional[Partition]]:
        if seq is None:
            return [fill] * target_len
        data = list(seq)
        if len(data) == target_len:
            return data
        if logger is not None:
            logger.warning(
                f"{label} length {len(data)} != expected {target_len}; normalizing"
            )
        if len(data) < target_len:
            return data + [fill] * (target_len - len(data))
        return data[:target_len]

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
            self.active_changing_split_tracking.append(None)
            # subtree tracking removed

            # Process tree pair (work on copies to avoid side effects)
            t0 = time.perf_counter()
            pre_sol = None
            if self.precomputed_pair_solutions is not None and i < len(
                self.precomputed_pair_solutions
            ):
                pre_sol = self.precomputed_pair_solutions[i]
            pair_result: Any = self.pair_processor(
                tree_list[i].deep_copy(),
                tree_list[i + 1].deep_copy(),
                i,
                pre_sol,
            )
            pair_elapsed = time.perf_counter() - t0

            self.logger.debug(
                f"Processed T{i}→T{i + 1} in {pair_elapsed:.3f}s; generated {len(pair_result.trees)} trees"
            )

            # Collect results into stateful attributes
            self.results.extend(pair_result.trees)
            num_pair_trees = len(pair_result.trees)
            self.mappings_one.append(pair_result.mapping_one)
            self.mappings_two.append(pair_result.mapping_two)

            # Normalize s-edge tracking to align 1:1 with trees
            track_list = self._normalize_len(
                getattr(pair_result, "active_changing_split_tracking", None),
                num_pair_trees,
                fill=None,
                label="s_edge_tracking",
                logger=self.logger,
            )

            self.active_changing_split_tracking.extend(track_list)
            # Count total generated trees per pair for reporting
            self.pair_interpolated_tree_counts.append(num_pair_trees)

            self.lattice_solutions_list.append(pair_result.lattice_edge_solutions)
            # distances removed

        # Add the final original tree to complete the sequence
        self.results.append(tree_list[-1])
        self.active_changing_split_tracking.append(None)
        # subtree tracking removed
        self.logger.info(
            f"Completed interpolation sequence: {len(self.results)} total trees from {len(tree_list)} originals + {sum(self.pair_interpolated_tree_counts)} interpolated"
        )

        self.last_result = TreeInterpolationSequence(
            interpolated_trees=self.results,
            mapping_one=self.mappings_one,
            mapping_two=self.mappings_two,
            active_changing_split_tracking=self.active_changing_split_tracking,
            pair_interpolated_tree_counts=self.pair_interpolated_tree_counts,
            lattice_solutions_list=self.lattice_solutions_list,
        )
        return self.last_result


def build_sequential_lattice_interpolations(
    tree_list: List[Node],
    precomputed_pair_solutions: Optional[
        List[Optional[Dict[Partition, List[List[Partition]]]]]
    ] = None,
) -> TreeInterpolationSequence:
    """Backwards-compatible wrapper that uses the stateful builder."""
    return SequentialInterpolationBuilder(
        precomputed_pair_solutions=precomputed_pair_solutions
    ).build(tree_list)
