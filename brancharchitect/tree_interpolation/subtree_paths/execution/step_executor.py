from typing import Any, Dict, List, Optional, Tuple
from brancharchitect.tree import Node
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from ..planning import build_edge_plan
from .microsteps import build_microsteps_for_selection
import logging

# Suppress the diagnostics logger
logging.getLogger(
    "brancharchitect.tree_interpolation.subtree_paths.planning.diagnostics"
).setLevel(logging.WARNING)


def apply_stepwise_plan_for_edge(
    current_base_tree: Node,
    destination_tree: Node,
    current_pivot_edge: Partition,
    expand_paths_for_pivot_edge: Dict[Partition, PartitionSet[Partition]],
    collapse_paths_for_pivot_edge: Dict[Partition, PartitionSet[Partition]],
) -> Tuple[List[Node], List[Optional[Partition]], Node, List[Partition]]:
    """
    Executes the stepwise plan for one pivot edge (active-changing split) across all selections.
    Returns the generated trees/edges and the updated interpolation state.

    Args:
        current_base_tree: The current tree state
        destination_tree: The destination tree we're morphing toward
        current_pivot_edge: The pivot edge (active-changing split) being processed
        expand_paths_for_pivot_edge: Paths for partitions that will be expanded
        collapse_paths_for_pivot_edge: Paths for partitions that will be collapsed

    Returns:
        Tuple of (trees, edges, interpolation_state, subtree_tracker)
    """
    trees: List[Node] = []
    edges: List[Optional[Partition]] = []
    subtree_tracker: List[Partition] = []
    interpolation_state: Node = current_base_tree.deep_copy()

    selections: Dict[Partition, Dict[str, Any]] = build_edge_plan(
        expand_paths_for_pivot_edge,
        collapse_paths_for_pivot_edge,
        current_base_tree,
        destination_tree,
        current_pivot_edge=current_pivot_edge,
    )

    # Clean logging of selections
    logging.debug(f"\nSelections for edge {current_pivot_edge}:")
    for subtree, selection in selections.items():
        logging.debug(f"\nSubtree: {subtree}")

        if "collapse" in selection and "path_segment" in selection["collapse"]:
            collapse_paths = selection["collapse"]["path_segment"]
            logging.debug(f"  Collapse paths ({len(collapse_paths)}):")
            for path in collapse_paths:
                logging.debug(f"    - {path}")

        if "expand" in selection and "path_segment" in selection["expand"]:
            expand_paths = selection["expand"]["path_segment"]
            logging.debug(f"  Expand paths ({len(expand_paths)}):")
            for path in expand_paths:
                logging.debug(f"    - {path}")

    for _, (subtree, selection) in enumerate(selections.items(), start=1):
        # Add the subtree back to the selection for compatibility
        selection_with_subtree: Dict[str, Any] = {**selection, "subtree": subtree}
        step_trees, step_edges, interpolation_state, step_subtree_tracker = (
            build_microsteps_for_selection(
                interpolation_state=interpolation_state,
                destination_tree=destination_tree,
                current_pivot_edge=current_pivot_edge,
                selection=selection_with_subtree,
            )
        )
        trees.extend(step_trees)
        edges.extend(step_edges)
        subtree_tracker.extend(step_subtree_tracker)

    return trees, edges, interpolation_state, subtree_tracker
