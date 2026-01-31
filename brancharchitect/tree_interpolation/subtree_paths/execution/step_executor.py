from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import logging

from brancharchitect.tree import Node
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree_interpolation.topology_ops.collapse import (
    create_collapsed_consensus_tree,
)
from brancharchitect.tree_interpolation.topology_ops.weights import (
    apply_zero_branch_lengths,
)
from brancharchitect.tree_interpolation.topology_ops.expand import (
    create_subtree_grafted_tree,
)
from ..planning import build_edge_plan
from .reordering import reorder_tree_toward_destination, align_to_source_order

logger = logging.getLogger(__name__)


# ============================================================================
# Weight Application Functions
# ============================================================================


def apply_expand_split_weights(
    tree: Node,
    expand_splits: List[Partition],
    destination_weights: Dict[Partition, float],
) -> None:
    """
    Set expand splits (new in destination) to their destination weight.

    Expand splits don't exist in the source tree, so there's nothing to average -
    we simply set them to their final destination weight.

    Args:
        tree: The tree to modify (mutated in place)
        expand_splits: List of splits that are new in the destination
        destination_weights: Weights from the destination tree
    """
    for split in expand_splits:
        node = tree.find_node_by_split(split)
        if node is not None:
            node.length = destination_weights.get(split, 0.0)


def apply_shared_split_average(
    tree: Node,
    split: Partition,
    source_weights: Optional[Dict[Partition, float]],
    destination_weights: Dict[Partition, float],
) -> None:
    """
    Average a shared split's weight: (source + dest) / 2.

    Shared splits exist in both source and destination, so we interpolate
    by taking the arithmetic mean of both weights.

    Args:
        tree: The tree to modify (mutated in place)
        split: The shared split to average
        source_weights: Weights from the original source tree
        destination_weights: Weights from the destination tree
    """
    node = tree.find_node_by_split(split)
    if node is not None:
        src_weight = (
            source_weights.get(split, node.length or 0.0)
            if source_weights
            else (node.length or 0.0)
        )
        dest_weight = destination_weights.get(split, 0.0)
        node.length = (src_weight + dest_weight) / 2.0


def apply_shared_splits_under_pivot(
    tree: Node,
    pivot_node: Node,
    expand_set: set[Partition],
    source_weights: Optional[Dict[Partition, float]],
    destination_weights: Dict[Partition, float],
) -> None:
    """
    Apply weight averaging to all shared splits under the pivot.

    For the first mover, we average ALL shared splits under the pivot edge.
    Expand splits are set to destination weight, shared splits are averaged.

    Args:
        tree: The tree to modify (mutated in place)
        pivot_node: The node representing the pivot edge
        expand_set: Set of splits that are expand (new) splits
        source_weights: Weights from the original source tree
        destination_weights: Weights from the destination tree
    """
    splits_under_pivot = list(pivot_node.to_splits(with_leaves=True))

    for split in splits_under_pivot:
        if split in expand_set:
            # Expand split: set to destination weight
            apply_expand_split_weights(tree, [split], destination_weights)
        else:
            # Shared split: average (source + dest) / 2
            apply_shared_split_average(tree, split, source_weights, destination_weights)


def apply_snap_weights(
    tree: Node,
    current_pivot_edge: Partition,
    expand_path: List[Partition],
    is_first_mover: bool,
    source_weights: Optional[Dict[Partition, float]],
    destination_weights: Dict[Partition, float],
) -> None:
    """
    Apply final weights during the snap phase.

    Weight application strategy:
    - First mover: Average all shared splits under pivot, set expand splits to dest weight
    - Subsequent movers: Only set their expand splits to destination weight
                         (shared splits were already averaged by first mover)

    Args:
        tree: The tree to modify (mutated in place)
        current_pivot_edge: The pivot edge being processed
        expand_path: List of splits to expand for this mover
        is_first_mover: Whether this is the first mover for this pivot
        source_weights: Weights from the original source tree
        destination_weights: Weights from the destination tree
    """
    pivot_node = tree.find_node_by_split(current_pivot_edge)

    if is_first_mover:
        # First mover: Average all shared splits under pivot + pivot edge itself
        if pivot_node is not None:
            expand_set = set(expand_path)
            apply_shared_splits_under_pivot(
                tree, pivot_node, expand_set, source_weights, destination_weights
            )
    else:
        # Subsequent movers: Only set expand splits to destination weight
        # Shared splits were already averaged by the first mover
        apply_expand_split_weights(tree, expand_path, destination_weights)


# ============================================================================
# Microstep Building Functions
# ============================================================================


def find_siblings_with_shared_parent_in_path(
    current_mover: Partition,
    current_path: List[Partition],
    all_movers: List[Partition],
    parent_map: Optional[Dict[Partition, Partition]],
) -> List[Partition]:
    """
    Identify sibling movers that share the same parent in the source/dest tree,
    ONLY if that parent is also present in the current mover's path.

    This ensures that when a parent node moves (is in the path), all its children
    (the siblings) are tracked and moved together in the same microstep group.
    """
    if not parent_map:
        return [current_mover]

    my_parent = parent_map.get(current_mover)

    # Critical Check: The parent MUST be in the path for this grouping to apply.
    # If the parent isn't moving (not in path), then these siblings are moving independently
    # through a static parent structure, so they should NOT be grouped.
    if not my_parent or my_parent not in current_path:
        return [current_mover]

    # Find all movers that share this same parent
    siblings = [
        m for m in all_movers if m != current_mover and parent_map.get(m) == my_parent
    ]

    # Return self + siblings, sorted for deterministic behavior
    group = [current_mover] + siblings
    # Sort by size first (descending) then bitmask for stability
    return sorted(group, key=lambda p: (-p.size, p.bitmask))


def _add_step(
    trees: List[Node],
    edges: List[Optional[Partition]],
    tree: Node,
    edge: Optional[Partition],
    subtree: Partition,
    subtree_tracker: Optional[List[List[Partition]]] = None,  # Updated type
    partition_group: Optional[List[Partition]] = None,  # New argument
) -> None:
    """Add a tree and edge to the collection of microsteps.

    Args:
        trees: List to append the tree to
        edges: List to append the edge to
        tree: The tree to add
        edge: The edge to add
        subtree: The primary subtree partition for this step (legacy)
        subtree_tracker: Optional list to track subtree partitions per step
        partition_group: Optional list of all partitions moving in this step
    """
    trees.append(tree)
    edges.append(edge)
    if subtree_tracker is not None:
        # If a group is provided (shared parent case), track the whole group.
        # Otherwise track the single subtree as a list of 1.
        if partition_group is not None:
            subtree_tracker.append(partition_group)
        else:
            subtree_tracker.append([subtree])


def build_microsteps_for_selection(
    interpolation_state: Node,
    destination_tree: Node,
    current_pivot_edge: Partition,
    selection: Dict[str, Any],
    all_mover_partitions: Optional[List[Partition]] = None,
    source_parent_map: Optional[Dict[Partition, Partition]] = None,
    dest_parent_map: Optional[Dict[Partition, Partition]] = None,
    is_first_mover: bool = True,
    is_last_mover: bool = True,
    source_weights: Optional[Dict[Partition, float]] = None,
    source_tree: Optional[Node] = None,
    step_progress: float = 0.5,
) -> Tuple[
    List[Node], List[Optional[Partition]], Node, List[List[Partition]]
]:  # Updated return type
    """
    Build the 5 microsteps for a single selection under an active-changing edge.

    Steps:
    - IT_down: set branch lengths to zero inside the subtree selection
    - C: collapse zero-length branches to consensus topology
    - C_reorder: partially reorder to match the destination
    - IT_up: graft the reference path while preserving order
    - IT_ref: apply final reference weights on the grafted path

    Args:
        interpolation_state: The current tree state before applying this selection's steps.
        destination_tree: The final target tree (used for weight lookups and consensus checks).
        current_pivot_edge: The active-changing split (pivot edge) currently being processed.
        selection: A dictionary containing the 'subtree' partition and its 'expand'/'collapse' paths.
        all_mover_partitions: List of all moving subtree Partitions (blocks) for this pivot edge.
                              Used to identify stable anchor blocks vs moving blocks.
        source_parent_map: Maps each mover -> its parent in source tree (from map_solution_elements_via_parent).
        dest_parent_map: Maps each mover -> its parent in destination tree (from map_solution_elements_via_parent).
        is_first_mover: Whether this is the first subtree being moved for the current pivot edge.
        is_last_mover: Whether this is the last subtree being moved for the current pivot edge.
        source_weights: Optional dictionary of original source weights for stable averaging.
        source_tree: Optional reference to the original source tree for tip distance compensation.
        step_progress: Current interpolation progress (0.0 = source, 1.0 = destination).
                       Used for progress-aware patristic distance compensation.

    Returns:
        Tuple containing:
        - List[Node]: The sequence of 5 intermediate trees generated by this selection.
        - List[Optional[Partition]]: The pivot edge associated with each step (for tracking).
        - Node: The final tree state after applying all microsteps (snapped_tree).
        - List[List[Partition]]: The subtree partitions (grouped) associated with each step.
    """
    trees: List[Node] = []
    edges: List[Optional[Partition]] = []
    subtree_tracker: List[List[Partition]] = []  # Updated type

    # Extract path segments directly from selection
    # Note: Paths are already filtered upstream in calculate_subtree_paths (transition_builder.py)
    # which removes pivot_edge from both paths and subtree from collapse path
    subtree_partition = selection["subtree"]
    expand_path: List[Partition] = selection.get("expand", {}).get("path_segment", [])
    zeroing_path: List[Partition] = selection.get("collapse", {}).get(
        "path_segment", []
    )

    # Log the paths being applied for debugging
    logger.debug(
        f"[StepExecutor] Subtree {subtree_partition}: "
        f"collapse_path={[tuple(s.indices) for s in zeroing_path]}, "
        f"expand_path={[tuple(s.indices) for s in expand_path]}"
    )

    # Determine movers for the collapse phase (Source side)
    collapse_movers = find_siblings_with_shared_parent_in_path(
        current_mover=subtree_partition,
        current_path=zeroing_path,
        all_movers=all_mover_partitions or [],
        parent_map=source_parent_map,  # Source context for collapse
    )

    _add_step(
        trees,
        edges,
        interpolation_state,
        current_pivot_edge,
        subtree_partition,  # Legacy single support
        subtree_tracker,
        partition_group=collapse_movers,  # New grouped support
    )

    # Step 1: Collapse Down - zero branch lengths
    it_down: Node = interpolation_state.deep_copy()
    # Move internal mass to pendants before zeroing (Causal Mass Transfer)
    # distribute_path_weights(it_down, zeroing_path, operation="add")

    # Then zero the branch lengths within the subtree selection
    apply_zero_branch_lengths(it_down, PartitionSet(set(zeroing_path)))

    _add_step(
        trees,
        edges,
        it_down,
        current_pivot_edge,
        subtree_partition,
        subtree_tracker,
        partition_group=collapse_movers,  # Use collapse group
    )

    logger.debug(f"[StepExecutor] After IT_Down: {len(it_down.to_splits())} splits")

    # Step 2: Collapse - merge zero-length branches
    # CRITICAL: Pass destination_tree to preserve common splits that exist in destination
    collapsed: Node = create_collapsed_consensus_tree(
        it_down, current_pivot_edge, copy=True, destination_tree=destination_tree
    )

    _add_step(
        trees,
        edges,
        collapsed,
        current_pivot_edge,
        subtree_partition,
        subtree_tracker,
        partition_group=collapse_movers,  # Use collapse group
    )

    logger.debug(f"[StepExecutor] After Collapse: {len(collapsed.to_splits())} splits")

    # Step 2.5: Tip Distance Compensation (Post-Collapse)
    # Not needed here as "collapsed" inherits lengths from "it_down"
    # which is already compensated in Step 1.5.

    # Step 3: Reorder - place subtree at new position
    reordered: Node = reorder_tree_toward_destination(
        source_tree=collapsed,
        destination_tree=destination_tree,
        current_pivot_edge=current_pivot_edge,
        moving_subtree_partition=subtree_partition,
        all_mover_partitions=all_mover_partitions,
        source_parent_map=source_parent_map,
        dest_parent_map=dest_parent_map,
        copy=True,
    )

    # For reorder step, use collapse_movers as we are still in "collapsed" state technically?
    # Actually, reordering is the bridge. Let's stick with collapse_movers for now as
    # visual consistency with previous steps is usually desired.
    _add_step(
        trees,
        edges,
        reordered,
        current_pivot_edge,
        subtree_partition,
        subtree_tracker,
        partition_group=collapse_movers,
    )

    logger.debug(f"[StepExecutor] After Reorder: {len(reordered.to_splits())} splits")

    destination_weights: Dict[Partition, float] = destination_tree.to_weighted_splits()

    reordered_mean: Node = reordered.deep_copy()

    # Calculate movers for Expand phase (Destination side)
    expand_movers = find_siblings_with_shared_parent_in_path(
        current_mover=subtree_partition,
        current_path=expand_path,
        all_movers=all_mover_partitions or [],
        parent_map=dest_parent_map,  # Destination context for expand
    )

    # Step 4: Expand Up - graft new branches onto the mean tree
    # New branches will have length 0, effectively separating Expansion from Mean calc.
    logger.debug(
        f"[StepExecutor] Subtree {subtree_partition} expanding {len(expand_path)} splits"
    )
    if len(expand_path) > 0:
        logger.debug(f"  Expand details: {[tuple(s.indices) for s in expand_path]}")

    pre_snap_reordered: Node = create_subtree_grafted_tree(
        base_tree=reordered_mean,
        ref_path_to_build=expand_path,
        copy=True,
    )

    pre_snap_reordered.reorder_taxa(list(reordered.get_current_order()))

    # Use align_to_source_order to preserve non-mover positions after graft
    align_to_source_order(
        pre_snap_reordered,
        source_order=list(reordered.get_current_order()),
        moving_taxa=subtree_partition.taxa,
    )

    # Align ordering once after graft so both microsteps share identical layout
    snapped_tree: Node = pre_snap_reordered.deep_copy()

    # Normalize to final snapped ordering
    final_order = list(snapped_tree.get_current_order())
    pre_snap_reordered.reorder_taxa(final_order)

    reordered.reorder_taxa(list(pre_snap_reordered.get_current_order()))

    # Step 5: Snap - apply weights based on split type
    # - Shared splits (in both source and dest): Average ONCE on first mover only
    # - Expand splits (new in dest): Set to destination weight
    # - Pivot edge: Average on first mover only

    apply_snap_weights(
        tree=snapped_tree,
        current_pivot_edge=current_pivot_edge,
        expand_path=expand_path,
        is_first_mover=is_first_mover,
        source_weights=source_weights,
        destination_weights=destination_weights,
    )

    _add_step(
        trees,
        edges,
        pre_snap_reordered,
        current_pivot_edge,
        subtree_partition,
        subtree_tracker,
        partition_group=expand_movers,  # Use expand group
    )

    _add_step(
        trees,
        edges,
        pre_snap_reordered,
        current_pivot_edge,
        subtree_partition,
        subtree_tracker,
        partition_group=expand_movers,  # Use expand group
    )

    logger.debug(
        f"[StepExecutor] After Expand: {len(pre_snap_reordered.to_splits())} splits"
    )

    _add_step(
        trees,
        edges,
        snapped_tree,
        current_pivot_edge,
        subtree_partition,
        subtree_tracker,
        partition_group=expand_movers,  # Use expand group
    )

    _add_step(
        trees,
        edges,
        snapped_tree,
        current_pivot_edge,
        subtree_partition,
        subtree_tracker,
        partition_group=expand_movers,  # Use expand group
    )

    logger.debug(f"[StepExecutor] After Snap: {len(snapped_tree.to_splits())} splits")

    return trees, edges, snapped_tree, subtree_tracker


# ============================================================================
# Edge Plan Execution
# ============================================================================


def apply_stepwise_plan_for_edge(
    current_base_tree: Node,
    destination_tree: Node,
    source_tree: Node,
    current_pivot_edge: Partition,
    expand_paths_for_pivot_edge: Dict[Partition, PartitionSet[Partition]],
    collapse_paths_for_pivot_edge: Dict[Partition, PartitionSet[Partition]],
    source_parent_map: Optional[Dict[Partition, Partition]] = None,
    dest_parent_map: Optional[Dict[Partition, Partition]] = None,
) -> Tuple[List[Node], List[Optional[Partition]], Node, List[List[Partition]]]:
    """
    Executes the stepwise plan for one pivot edge across all selections.

    Args:
        current_base_tree: The current tree state (interpolation state)
        destination_tree: The destination tree we're morphing toward
        source_tree: The ORIGINAL source tree (for split computation)
        current_pivot_edge: The pivot edge (active-changing split) being processed
        expand_paths_for_pivot_edge: Paths for partitions that will be expanded
        collapse_paths_for_pivot_edge: Paths for partitions that will be collapsed
        source_parent_map: Maps each mover -> its parent in source tree
        dest_parent_map: Maps each mover -> its parent in destination tree
        edge_progress_start: Progress value at the start of this edge's processing (0.0-1.0)
        edge_progress_end: Progress value at the end of this edge's processing (0.0-1.0)

    Returns:
        Tuple of (trees, edges, interpolation_state, subtree_tracker)
    """
    trees: List[Node] = []
    edges: List[Optional[Partition]] = []
    subtree_tracker: List[List[Partition]] = []
    interpolation_state: Node = current_base_tree

    selections: Dict[Partition, Dict[str, Any]] = build_edge_plan(
        expand_paths_for_pivot_edge,
        collapse_paths_for_pivot_edge,
        source_tree,  # Use original source tree for split computation, NOT interpolation state
        destination_tree,
        current_pivot_edge=current_pivot_edge,
    )

    # Calculate source weights once using the ORIGINAL source tree
    # This ensures consistent (Source + Dest) / 2 interpolation
    source_weights: Dict[Partition, float] = source_tree.to_weighted_splits()

    # All mover partitions as BLOCKS (not flattened to taxa)
    # CRITICAL: We must include ALL subtrees that have paths, even if they were dropped
    # from the plan (Passenger subtrees handled by Drivers).
    # Using selections.keys() would miss these passengers, causing split grouping logic to fail.
    all_mover_partitions: List[Partition] = list(
        set(list(expand_paths_for_pivot_edge.keys()))
        | set(list(collapse_paths_for_pivot_edge.keys()))
    )

    # We iterate over items to track first/last mover status
    selection_items = list(selections.items())
    total_selections = len(selection_items)

    last_subtree_partition: Optional[Partition] = None

    for i, (subtree, selection) in enumerate(selection_items):
        last_subtree_partition = subtree
        is_first_mover = i == 0
        is_last_mover = i == total_selections - 1

        # Add subtree to selection for compatibility
        selection_with_subtree: Dict[str, Any] = {**selection, "subtree": subtree}

        step_trees, step_edges, interpolation_state, step_subtree_tracker = (
            build_microsteps_for_selection(
                interpolation_state=interpolation_state,
                destination_tree=destination_tree,
                current_pivot_edge=current_pivot_edge,
                selection=selection_with_subtree,
                all_mover_partitions=all_mover_partitions,
                source_parent_map=source_parent_map,
                dest_parent_map=dest_parent_map,
                is_first_mover=is_first_mover,
                is_last_mover=is_last_mover,
                source_weights=source_weights,
                source_tree=source_tree,
            )
        )

        trees.extend(step_trees)
        edges.extend(step_edges)
        subtree_tracker.extend(step_subtree_tracker)

    # Add final destination state for this pivot edge as requested
    if last_subtree_partition is not None:
        _add_step(
            trees,
            edges,
            interpolation_state,
            current_pivot_edge,
            last_subtree_partition,
            subtree_tracker,
        )
    elif not trees:
        # No selections means no work to do for this pivot edge (reordering only).
        # Add the current state as a pass-through step so interpolation can continue.
        trees.append(interpolation_state.deep_copy())
        edges.append(current_pivot_edge)
        subtree_tracker.append([])

    return trees, edges, interpolation_state, subtree_tracker
