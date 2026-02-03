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


def set_expand_splits_to_dest_weight(
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


def average_weights_under_pivot(
    tree: Node,
    pivot_node: Node,
    expand_set: set[Partition],
    source_weights: Optional[Dict[Partition, float]],
    destination_weights: Dict[Partition, float],
) -> None:
    """
    Average branch weights for all splits under the pivot edge.

    For the first mover, we process ALL splits under the pivot:
    - Expand splits (new in destination): set to destination weight
    - Shared splits (in both trees): average (source + dest) / 2

    Args:
        tree: The tree to modify (mutated in place)
        pivot_node: The node representing the pivot edge
        expand_set: Set of splits that are expand (new) splits
        source_weights: Weights from the original source tree
        destination_weights: Weights from the destination tree
    """
    splits_under_pivot = list(pivot_node.to_splits(with_leaves=True))

    for split in splits_under_pivot:
        node = tree.find_node_by_split(split)
        if node is None:
            continue

        if split in expand_set:
            # Expand split: set to destination weight
            node.length = destination_weights.get(split, 0.0)
        else:
            # Shared split: average (source + dest) / 2
            src_weight = (
                source_weights.get(split, node.length or 0.0)
                if source_weights
                else (node.length or 0.0)
            )
            dest_weight = destination_weights.get(split, 0.0)
            node.length = (src_weight + dest_weight) / 2.0


def finalize_branch_weights(
    tree: Node,
    current_pivot_edge: Partition,
    expand_path: List[Partition],
    is_first_mover: bool,
    source_weights: Optional[Dict[Partition, float]],
    destination_weights: Dict[Partition, float],
) -> None:
    """
    Finalize branch weights during the snap phase.

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
            average_weights_under_pivot(
                tree, pivot_node, expand_set, source_weights, destination_weights
            )
    else:
        # Subsequent movers: Only set expand splits to destination weight
        # Shared splits were already averaged by the first mover
        set_expand_splits_to_dest_weight(tree, expand_path, destination_weights)


# ============================================================================
# Microstep Building Functions
# ============================================================================


def find_sibling_movers(
    current_mover: Partition,
    current_path: List[Partition],
    all_movers: List[Partition],
    parent_map: Optional[Dict[Partition, Partition]],
) -> List[Partition]:
    """
    Find sibling movers that share the same parent in the source/dest tree,
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


def _append_frame(
    trees: List[Node],
    edges: List[Optional[Partition]],
    tree: Node,
    edge: Optional[Partition],
    subtree: Partition,
    subtree_tracker: Optional[List[List[Partition]]] = None,
    partition_group: Optional[List[Partition]] = None,
) -> None:
    """Append an animation frame to the output lists.

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


def build_frames_for_subtree(
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
) -> Tuple[List[Node], List[Optional[Partition]], Node, List[List[Partition]]]:
    """
    Build animation frames for a single subtree/mover under an active-changing edge.

    The function performs 4 phases:
    1. Collapse: Zero branch lengths and remove zero-length branches
    2. Reorder: Move subtree to its destination position
    3. Expand: Graft new branches (with zero weights initially)
    4. Snap: Apply final branch weights

    Only phases with actual work generate animation frames. The returned
    final_state is always a fresh copy for safe chaining to the next subtree.

    Args:
        interpolation_state: The current tree state before applying this selection's steps.
        destination_tree: The final target tree (used for weight lookups and consensus checks).
        current_pivot_edge: The active-changing split (pivot edge) currently being processed.
        selection: A dictionary containing the 'subtree' partition and its 'expand'/'collapse' paths.
        all_mover_partitions: List of all moving subtree Partitions (blocks) for this pivot edge.
        source_parent_map: Maps each mover -> its parent in source tree.
        dest_parent_map: Maps each mover -> its parent in destination tree.
        is_first_mover: Whether this is the first subtree being moved for the current pivot edge.
        is_last_mover: Whether this is the last subtree being moved for the current pivot edge.
        source_weights: Optional dictionary of original source weights for stable averaging.
        source_tree: Optional reference to the original source tree.
        step_progress: Current interpolation progress (0.0 = source, 1.0 = destination).

    Returns:
        Tuple containing:
        - List[Node]: Animation frames (intermediate trees) for this selection.
        - List[Optional[Partition]]: The pivot edge associated with each frame.
        - Node: The final tree state after all transformations (always a fresh copy).
        - List[List[Partition]]: The subtree partitions (grouped) associated with each frame.
    """
    trees: List[Node] = []
    edges: List[Optional[Partition]] = []
    subtree_tracker: List[List[Partition]] = []

    # =========================================================================
    # Extract paths and detect work
    # =========================================================================
    subtree_partition = selection["subtree"]
    collapse_paths: List[Partition] = selection.get("collapse", {}).get(
        "path_segment", []
    )
    expand_paths: List[Partition] = selection.get("expand", {}).get("path_segment", [])

    has_collapse_work = len(collapse_paths) > 0
    has_expand_work = len(expand_paths) > 0

    logger.debug(
        f"[StepExecutor] Subtree {subtree_partition}: "
        f"collapse_paths={[tuple(s.indices) for s in collapse_paths]}, "
        f"expand_paths={[tuple(s.indices) for s in expand_paths]}, "
        f"is_first_mover={is_first_mover}"
    )

    # =========================================================================
    # Determine mover groups for tracking
    # =========================================================================
    collapse_movers = find_sibling_movers(
        current_mover=subtree_partition,
        current_path=collapse_paths,
        all_movers=all_mover_partitions or [],
        parent_map=source_parent_map,
    )
    expand_movers = find_sibling_movers(
        current_mover=subtree_partition,
        current_path=expand_paths,
        all_movers=all_mover_partitions or [],
        parent_map=dest_parent_map,
    )

    # =========================================================================
    # Phase 1: Collapse (compute always, add frames conditionally)
    # =========================================================================
    if has_collapse_work:
        zeroed_tree: Node = interpolation_state.deep_copy()
        apply_zero_branch_lengths(zeroed_tree, PartitionSet(set(collapse_paths)))
        collapsed_tree: Node = create_collapsed_consensus_tree(
            zeroed_tree,
            current_pivot_edge,
            copy=True,
            destination_tree=destination_tree,
        )

        # Add collapse frames
        _append_frame(
            trees,
            edges,
            interpolation_state.deep_copy(),
            current_pivot_edge,
            subtree_partition,
            subtree_tracker,
            partition_group=collapse_movers,
        )
        _append_frame(
            trees,
            edges,
            zeroed_tree,
            current_pivot_edge,
            subtree_partition,
            subtree_tracker,
            partition_group=collapse_movers,
        )
        _append_frame(
            trees,
            edges,
            collapsed_tree.deep_copy(),
            current_pivot_edge,
            subtree_partition,
            subtree_tracker,
            partition_group=collapse_movers,
        )

        logger.debug(
            f"[StepExecutor] After Collapse: {len(collapsed_tree.to_splits())} splits"
        )
    else:
        # No collapse work - start from a copy of input
        collapsed_tree = interpolation_state.deep_copy()

    # =========================================================================
    # Phase 2: Reorder (compute always, add frames conditionally)
    # =========================================================================
    reordered_tree: Node = reorder_tree_toward_destination(
        source_tree=collapsed_tree,
        destination_tree=destination_tree,
        current_pivot_edge=current_pivot_edge,
        moving_subtree_partition=subtree_partition,
        all_mover_partitions=all_mover_partitions,
        source_parent_map=source_parent_map,
        dest_parent_map=dest_parent_map,
        copy=True,
    )

    # Detect if reorder actually changed anything (returns same object if no change)
    has_reorder_change = reordered_tree is not collapsed_tree

    if has_reorder_change:
        # Add reorder frames: before and after
        if not has_collapse_work:
            # Need to show pre-reorder state (collapsed_tree wasn't added yet)
            _append_frame(
                trees,
                edges,
                collapsed_tree.deep_copy(),
                current_pivot_edge,
                subtree_partition,
                subtree_tracker,
                partition_group=collapse_movers,
            )
        _append_frame(
            trees,
            edges,
            reordered_tree.deep_copy(),
            current_pivot_edge,
            subtree_partition,
            subtree_tracker,
            partition_group=collapse_movers,
        )
        logger.debug(
            f"[StepExecutor] After Reorder: {len(reordered_tree.to_splits())} splits"
        )
    else:
        # No reorder change - ensure we have a copy for chaining
        reordered_tree = collapsed_tree.deep_copy()

    # =========================================================================
    # Early exit if no further work
    # =========================================================================
    if not is_first_mover and not has_expand_work:
        # No expand or snap work - return reordered state
        if not trees:
            # No frames at all - still return a copy
            return [], [], reordered_tree, []
        return trees, edges, reordered_tree, subtree_tracker

    # =========================================================================
    # Phase 3: Expand/Graft (compute and add frames conditionally)
    # =========================================================================
    destination_weights: Dict[Partition, float] = destination_tree.to_weighted_splits()

    if has_expand_work:
        grafted_zero_weights: Node = create_subtree_grafted_tree(
            base_tree=reordered_tree,
            ref_path_to_build=expand_paths,
            copy=True,
        )

        # Restore ordering after graft
        grafted_zero_weights.reorder_taxa(list(reordered_tree.get_current_order()))

        # Preserve non-mover positions after graft
        align_to_source_order(
            grafted_zero_weights,
            source_order=list(reordered_tree.get_current_order()),
            moving_taxa=subtree_partition.taxa,
        )

        # Base for snap (copy before weights applied)
        grafted_tree: Node = grafted_zero_weights.deep_copy()

        # Capture final order BEFORE applying weights (more defensive)
        # Grafting may introduce new ordering that collapsed consensus doesn't have
        final_order = list(grafted_tree.get_current_order())

        # Normalize all earlier frames to match the grafted ordering
        # This ensures consistent leaf order across collapse -> reorder -> expand -> snap
        grafted_zero_weights.reorder_taxa(final_order)
        reordered_tree.reorder_taxa(final_order)

        # Apply weights to snap tree (doesn't change ordering)
        finalize_branch_weights(
            tree=grafted_tree,
            current_pivot_edge=current_pivot_edge,
            expand_path=expand_paths,
            is_first_mover=is_first_mover,
            source_weights=source_weights,
            destination_weights=destination_weights,
        )

        # Add expand frame (grafted with zero weights)
        _append_frame(
            trees,
            edges,
            grafted_zero_weights.deep_copy(),
            current_pivot_edge,
            subtree_partition,
            subtree_tracker,
            partition_group=expand_movers,
        )

        logger.debug(
            f"[StepExecutor] After Expand: {len(grafted_zero_weights.to_splits())} splits"
        )
    else:
        # No expand work - snap operates on reordered tree directly
        # (reordered_tree is already a fresh copy we own)
        grafted_tree = reordered_tree

        # Apply weights to snap tree
        finalize_branch_weights(
            tree=grafted_tree,
            current_pivot_edge=current_pivot_edge,
            expand_path=expand_paths,
            is_first_mover=is_first_mover,
            source_weights=source_weights,
            destination_weights=destination_weights,
        )

    # =========================================================================
    # Phase 4: Snap frame
    # =========================================================================
    _append_frame(
        trees,
        edges,
        grafted_tree.deep_copy(),
        current_pivot_edge,
        subtree_partition,
        subtree_tracker,
        partition_group=expand_movers,
    )

    logger.debug(f"[StepExecutor] After Snap: {len(grafted_tree.to_splits())} splits")

    return trees, edges, grafted_tree, subtree_tracker


# ============================================================================
# Edge Plan Execution
# ============================================================================


def execute_pivot_edge_plan(
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
    Execute the interpolation plan for one pivot edge across all subtrees.

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

    for i, (subtree, selection) in enumerate(selection_items):
        is_first_mover = i == 0
        is_last_mover = i == total_selections - 1

        # Add subtree to selection for compatibility
        selection_with_subtree: Dict[str, Any] = {**selection, "subtree": subtree}

        step_trees, step_edges, interpolation_state, step_subtree_tracker = (
            build_frames_for_subtree(
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

    # Handle edge case: no selections means no work to do for this pivot edge
    if not trees:
        # Add the current state as a pass-through step so interpolation can continue.
        trees.append(interpolation_state.deep_copy())
        edges.append(current_pivot_edge)
        subtree_tracker.append([])

    return trees, edges, interpolation_state, subtree_tracker
