from typing import Any, Dict, List, Optional, Tuple
from brancharchitect.tree import Node
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from ..planning import build_edge_plan
from .microsteps import build_microsteps_for_selection


def apply_stepwise_plan_for_edge(
    current_base_tree: Node,
    reference_tree: Node,
    reference_weights: Dict[Partition, float],
    active_changing_edge: Partition,
    expand_paths_for_s_edge: Dict[Partition, PartitionSet[Partition]],
    collapse_paths_for_s_edge: Dict[Partition, PartitionSet[Partition]],
    tree_index: int,
    active_changing_edge_ordinal: int,
) -> Tuple[
    List[Node], List[str], List[Optional[Partition]], Node, List[Optional[Partition]]
]:
    """
    Executes the stepwise plan for one s-edge across all selections.
    Returns the generated trees/names/edges and the updated interpolation state.

    Args:
        current_base_tree: The current tree state
        reference_tree: The target reference tree
        reference_weights: Weights for the reference tree partitions
        active_changing_edge: The edge being processed
        expand_paths_for_s_edge: Paths for partitions that will be expanded
        collapse_paths_for_s_edge: Paths for partitions that will be collapsed
        tree_index: Index of the current tree in the sequence
        active_changing_edge_ordinal: Ordinal of the active changing edge

    Returns:
        Tuple of (trees, names, edges, interpolation_state, subtree_tracking)
    """
    trees: List[Node] = []
    names: List[str] = []
    edges: List[Optional[Partition]] = []
    subtree_tracking: List[Optional[Partition]] = []
    interpolation_state: Node = current_base_tree.deep_copy()

    selections: Dict[Partition, Dict[str, Any]] = build_edge_plan(
        expand_paths_for_s_edge,
        collapse_paths_for_s_edge,
        current_base_tree,
        reference_tree,
        active_changing_edge,
    )

    for step_idx, (subtree, selection) in enumerate(selections.items(), start=1):
        # Add the subtree back to the selection for compatibility
        selection_with_subtree: Dict[str, Any] = {**selection, "subtree": subtree}
        step_trees, step_names, step_edges, interpolation_state, step_subtree_tracking = (
            build_microsteps_for_selection(
                interpolation_state=interpolation_state,
                reference_tree=reference_tree,
                reference_weights=reference_weights,
                active_changing_edge=active_changing_edge,
                selection=selection_with_subtree,
                tree_index=tree_index,
                active_changing_edge_ordinal=active_changing_edge_ordinal,
                step_idx=step_idx,
            )
        )
        trees.extend(step_trees)
        names.extend(step_names)
        edges.extend(step_edges)
        subtree_tracking.extend(step_subtree_tracking)

    return trees, names, edges, interpolation_state, subtree_tracking
