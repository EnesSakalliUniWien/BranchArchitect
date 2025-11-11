"""Sequence orchestrator for subtree-path interpolation.

Coordinates the high-level interpolation sequence for each pivot edge (current_pivot_edge)
split by:
- Calculating per-split subtree paths
- Planning collapse/expand steps per subtree
- Executing the 5 microsteps and aggregating results

Terminology:
    - pivot_edge / current_pivot_edge: The split being processed (preferred term)
    - active-changing split: Formal documentation term (same concept)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional
from brancharchitect.tree import Node
from brancharchitect.elements.partition import Partition
from .paths import calculate_subtree_paths
from .execution.step_executor import apply_stepwise_plan_for_edge

logger = logging.getLogger(__name__)


def create_interpolation_for_active_split_sequence(
    source_tree: Node,
    destination_tree: Node,
    target_pivot_edges: List[Partition],
    jumping_subtree_solutions: Dict[Partition, List[Partition]],
) -> tuple[
    List[Node],
    List[Partition],
    List[Optional[Partition]],
]:
    """
    Create an interpolation sequence from source to destination tree for pivot edges (active-changing splits).

    Strategy (modular and stepwise):
      1) Calculate subtree paths once for destination and source.
      2) For each current_pivot_edge on the current interpolation state:
         - If the current_pivot_edge is missing in either tree, run classical fallback.
         - Else, try stepwise plan:
             a) Iterate selections (individual/whole per rule).
             b) Apply micro-steps per selection.
           If no selections are produced, run the simple 5-step fallback for that current_pivot_edge.
    """
    interpolation_sequence: List[Node] = []
    failed_pivot_edges: List[Partition] = []
    processed_pivot_edge_tracking: List[Optional[Partition]] = []
    # subtree tracking removed

    interpolation_state: Node = source_tree.deep_copy()

    # Precompute subtree paths in destination/source
    destination_subtree_paths, source_subtree_paths = calculate_subtree_paths(
        jumping_subtree_solutions, destination_tree, source_tree
    )

    current_base_tree = interpolation_state.deep_copy()

    for pivot_idx, current_pivot_edge in enumerate(target_pivot_edges):
        logger.debug(
            f"[ORCHESTRATOR] Pivot {pivot_idx + 1}/{len(target_pivot_edges)}: {current_pivot_edge.bipartition()}"
        )

        current_base_tree: Node = interpolation_state.deep_copy()

        current_base_tree.initialize_split_indices(current_base_tree.taxa_encoding)

        # Paths for this current_pivot_edge (kept separate for destination/source)
        # Keep as PartitionSet[Partition] - no conversion needed
        source_paths_for_pivot_edge = source_subtree_paths.get(current_pivot_edge, {})
        destination_paths_for_pivot_edge = destination_subtree_paths.get(
            current_pivot_edge, {}
        )

        logger.debug(
            f"[ORCHESTRATOR] Paths src={len(source_paths_for_pivot_edge)} dst={len(destination_paths_for_pivot_edge)}"
        )

        # Guard: verify the pivot edge exists in both trees before planning
        logger.debug("[ORCHESTRATOR] Checking pivot presence in both trees...")

        src_node = current_base_tree.find_node_by_split(current_pivot_edge)
        dst_node = destination_tree.find_node_by_split(current_pivot_edge)

        logger.debug(
            f"[ORCHESTRATOR] Present src={src_node is not None} dst={dst_node is not None}"
        )

        if src_node is None or dst_node is None:
            # Analyze why this pivot edge is missing
            missing_in: list[str] = []
            if src_node is None:
                missing_in.append("source (current interpolation state)")
            if dst_node is None:
                missing_in.append("destination")

            raise ValueError(
                f"Pivot edge {current_pivot_edge.bipartition()} "
                f"missing in {' and '.join(missing_in)} tree. "
                f"This edge was identified by the lattice algorithm but doesn't exist in both trees. "
                f"See debug logs above for edge comparison."
            )

        logger.debug("[ORCHESTRATOR]   ✓ Pivot edge found in both trees, proceeding...")

        step_trees, step_edges, new_state = apply_stepwise_plan_for_edge(
            current_base_tree=current_base_tree,
            destination_tree=destination_tree,
            current_pivot_edge=current_pivot_edge,
            expand_paths_for_pivot_edge=destination_paths_for_pivot_edge,
            collapse_paths_for_pivot_edge=source_paths_for_pivot_edge,
        )

        if step_trees:
            interpolation_sequence.extend(step_trees)
            processed_pivot_edge_tracking.extend(step_edges)

            # DEBUG: Check if new_state topology matches what we expect
            new_state_node_count = len(new_state.traverse())
            destination_node_count = len(destination_tree.traverse())

            logger.debug(
                f"[ORCHESTRATOR] Pivot {pivot_idx + 1} done: new={new_state_node_count} dst={destination_node_count} nodes"
            )

            interpolation_state = new_state

            if new_state_node_count != destination_node_count:
                new_state_splits = new_state.to_splits()
                destination_splits = destination_tree.to_splits()

                if new_state_splits == destination_splits:
                    logger.debug(
                        "[ORCHESTRATOR]     Node count differs but topology matches (likely due to unary-node collapse)."
                    )
                else:
                    logger.warning(
                        "[ORCHESTRATOR] Pivot %s topology mismatch (expected %s nodes, got %s)",
                        pivot_idx + 1,
                        destination_node_count,
                        new_state_node_count,
                    )

                    missing_partitions = destination_splits - new_state_splits
                    extra_partitions = new_state_splits - destination_splits

                    if missing_partitions:
                        logger.debug(
                            "[ORCHESTRATOR]     Missing splits: %s",
                            ", ".join(
                                str(partition.indices)
                                for partition in missing_partitions
                            ),
                        )

                    if extra_partitions:
                        logger.debug(
                            "[ORCHESTRATOR]     Extra splits: %s",
                            ", ".join(
                                str(partition.indices) for partition in extra_partitions
                            ),
                        )

    return (
        interpolation_sequence,
        failed_pivot_edges,
        processed_pivot_edge_tracking,
    )
