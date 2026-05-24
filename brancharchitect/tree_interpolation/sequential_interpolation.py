"""
Sequential lattice interpolation public API.

Provides the stateful builder and wrapper that construct sequential
interpolations across adjacent tree pairs to create smooth animations
between phylogenetic trees.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Dict, Callable

from brancharchitect.elements.partition import Partition
from brancharchitect.tree import Node
from brancharchitect.tree_interpolation.utils import iter_consecutive_pairs
from brancharchitect.tree_interpolation.types import (
    AttachmentEdgeMap,
    SprMoveEvent,
    TreeInterpolationSequence,
    TreePairInterpolation,
    build_attachment_edge_map,
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
        precomputed_lattice_solutions: Optional[
            List[Optional[Dict[Partition, List[Partition]]]]
        ] = None,
    ):
        """
        Initialize the sequential interpolation builder.

        Args:
            logger: Logger instance for operation tracking.
                If None, uses the module's default logger.
            precomputed_lattice_solutions: Pre-calculated lattice solutions for each pair.
                Can significantly speed up processing when available.
        """
        # Configure logging with fallback to module logger
        self.logger = logger or logging.getLogger(__name__)

        self.precomputed_lattice_solutions = precomputed_lattice_solutions
        self._initialize_build_state()

    def _initialize_build_state(self) -> None:
        """Initialize all state variables for a fresh interpolation build."""
        self.interpolated_trees: List[Node] = []
        self.attachment_edge_maps: List[AttachmentEdgeMap] = []
        self.active_pivot_edges: List[Optional[Partition]] = []
        self.current_subtree_highlights: List[Optional[List[Partition]]] = []
        self.spr_move_events: List[List[SprMoveEvent]] = []
        self.pair_tree_counts: List[int] = []
        self.affected_subtrees_by_split: List[Dict[Partition, List[Partition]]] = []

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

        if not (
            len(interpolation_result.trees)
            == len(interpolation_result.active_pivot_edges)
            == len(interpolation_result.current_subtree_highlights)
        ):
            raise RuntimeError("Interpolation result arrays are not aligned")

        landing_tree = (
            interpolation_result.trees[-1] if interpolation_result.trees else None
        )
        emitted_trees = interpolation_result.trees[:-1]
        emitted_pivot_edges = interpolation_result.active_pivot_edges[:-1]
        emitted_subtree_highlights = interpolation_result.current_subtree_highlights[
            :-1
        ]
        emitted_step_count = len(emitted_trees)

        # Collect transition frames only. The exact landing tree is represented
        # by the next input delimiter, not as an active interpolation frame.
        self.interpolated_trees.extend(emitted_trees)

        # Trees and tracking should have 1:1 correspondence from interpolation
        self.active_pivot_edges.extend(emitted_pivot_edges)
        self.current_subtree_highlights.extend(emitted_subtree_highlights)
        self.spr_move_events.append(
            _clip_spr_move_events(
                interpolation_result.spr_move_events,
                emitted_step_count - 1,
            )
        )

        self.pair_tree_counts.append(emitted_step_count)

        self.affected_subtrees_by_split.append(
            interpolation_result.jumping_subtree_solutions or {}
        )

        # Build and store source/destination attachment edges for this pair.
        if interpolation_result.jumping_subtree_solutions:
            source_map, destination_map = generate_solution_mappings(
                interpolation_result.jumping_subtree_solutions,
                destination=t2,
                source=t1,
            )
            self.attachment_edge_maps.append(
                build_attachment_edge_map(source_map, destination_map)
            )
        else:
            self.attachment_edge_maps.append({})

        # Return the final resolved tree from this interpolation
        if landing_tree is not None:
            return landing_tree

        # For identical trees (no interpolation), return the destination tree
        # aligned to the current source order. Child order is visual layout, not
        # topology; preserving it prevents false motion for no-op pairs.
        self.logger.debug(
            f"No interpolation needed for pair {pair_index} - returning destination tree aligned to source order"
        )
        aligned_destination = t2.deep_copy(build_split_index=False)
        aligned_destination.reorder_taxa(list(t1.get_current_order()))
        return aligned_destination

    def _add_delimiter_frame(self, tree: Node) -> None:
        """Add an original tree and a None highlight marker to the sequence as a delimiter."""
        # Deep copy to create an independent snapshot
        self.interpolated_trees.append(tree.deep_copy(build_split_index=False))
        self.active_pivot_edges.append(None)
        self.current_subtree_highlights.append(None)

    def _finalize_sequence(self, original_tree_count: int) -> TreeInterpolationSequence:
        """Construct the final sequence object and log a summary."""
        self.logger.info(
            f"Completed interpolation sequence: {len(self.interpolated_trees)} total trees from {original_tree_count} originals + {sum(self.pair_tree_counts)} interpolated"
        )

        return TreeInterpolationSequence(
            interpolated_trees=self.interpolated_trees,
            attachment_edge_maps=self.attachment_edge_maps,
            active_pivot_edges=self.active_pivot_edges,
            current_subtree_highlights=self.current_subtree_highlights,
            spr_move_events_list=self.spr_move_events,
            pair_interpolated_tree_counts=self.pair_tree_counts,
            affected_subtrees_by_split_list=self.affected_subtrees_by_split,
        )

    def build(
        self,
        trees: List[Node],
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> TreeInterpolationSequence:
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

        total_pairs = len(trees) - 1
        final_resolved_tree: Optional[Node] = None
        for pair in iter_consecutive_pairs(trees):
            pair_index, source, target, is_first, is_last = pair

            if progress_callback:
                pct = (pair_index / total_pairs) * 100
                progress_callback(
                    pct, f"Interpolating pair {pair_index + 1}/{total_pairs}"
                )

            precomputed_solution = (
                self.precomputed_lattice_solutions[pair_index]
                if self.precomputed_lattice_solutions is not None
                else None
            )
            source_tree = source if is_first else self.interpolated_trees[-1]

            # Process the pair and get the final resolved tree
            resolved_tree = self._process_pair(
                source_tree,
                target,
                pair_index,
                precomputed_solution,
            )
            final_resolved_tree = resolved_tree

            # Add the resolved tree as the delimiter for the next pair
            # This ensures: last frame of pair N = first frame of pair N+1
            if not is_last:
                self._add_delimiter_frame(resolved_tree)

        # Use the final resolved tree as the last delimiter to maintain sequence continuity
        if final_resolved_tree is None:
            raise RuntimeError("Interpolation did not process any tree pairs")
        self._add_delimiter_frame(final_resolved_tree)

        return self._finalize_sequence(len(trees))


def _clip_spr_move_events(
    events: List[SprMoveEvent],
    max_step_index: int,
) -> List[SprMoveEvent]:
    """Keep SPR event ranges aligned with emitted transition frames."""
    if max_step_index < 0:
        return []

    clipped: List[SprMoveEvent] = []
    for event in events:
        step_start, step_end = event["step_range"]
        if step_start > max_step_index:
            continue
        clipped.append(
            {
                **event,
                "step_range": (step_start, min(step_end, max_step_index)),
            }
        )
    return clipped
