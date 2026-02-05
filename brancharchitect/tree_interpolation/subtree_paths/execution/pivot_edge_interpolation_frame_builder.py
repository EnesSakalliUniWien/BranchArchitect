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
    finalize_branch_weights,
)

from brancharchitect.tree_interpolation.topology_ops.expand import (
    create_subtree_grafted_tree,
)
from ..planning import build_edge_plan
from .reordering import reorder_tree_toward_destination, align_to_source_order
from .sibling_grouping import (
    compute_sibling_groups,
    get_collapse_splits,
    get_expand_splits,
    get_group_for_mover,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Microstep Building Functions
# ============================================================================


def _append_frame(
    trees: List[Node],
    edges: List[Optional[Partition]],
    tree: Node,
    edge: Optional[Partition],
    subtree_tracker: List[List[Partition]],
    partition_group: List[Partition],
) -> None:
    """Append an animation frame to the output lists.

    Args:
        trees: List to append the tree to
        edges: List to append the edge to
        tree: The tree to add
        edge: The edge to add
        subtree_tracker: List to track subtree partitions per step
        partition_group: List of all partitions moving in this step
    """
    trees.append(tree)
    edges.append(edge)
    subtree_tracker.append(partition_group)


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
    collapse_sibling_groups: Optional[Dict[Partition, List[Partition]]] = None,
    expand_sibling_groups: Optional[Dict[Partition, List[Partition]]] = None,
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

    # =========================================================================
    # Determine mover groups for tracking (use pre-computed phase-specific groups)
    # =========================================================================
    # Collapse phase: use source parent grouping
    collapse_movers: List[Partition] = [subtree_partition]
    if collapse_sibling_groups:
        collapse_movers = get_group_for_mover(
            subtree_partition, collapse_sibling_groups
        )

    # Expand phase: use dest parent grouping
    expand_movers: List[Partition] = [subtree_partition]
    if expand_sibling_groups:
        expand_movers = get_group_for_mover(subtree_partition, expand_sibling_groups)

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

        _append_frame(
            trees,
            edges,
            zeroed_tree,
            current_pivot_edge,
            subtree_tracker,
            collapse_movers,
        )

        _append_frame(
            trees,
            edges,
            collapsed_tree.deep_copy(),
            current_pivot_edge,
            subtree_tracker,
            collapse_movers,
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
        source_parent_map=source_parent_map,
        dest_parent_map=dest_parent_map,
        all_mover_partitions=all_mover_partitions,
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
                subtree_tracker,
                collapse_movers,
            )

        _append_frame(
            trees,
            edges,
            reordered_tree.deep_copy(),
            current_pivot_edge,
            subtree_tracker,
            collapse_movers,
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

        grafted_zero_weights.reorder_taxa(list(reordered_tree.get_current_order()))

        # Use align_to_source_order to preserve non-mover positions after graft
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

        if trees:
            trees[-1].reorder_taxa(final_order)

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
            subtree_tracker,
            expand_movers,
        )

    else:
        # No expand work - snap operates on reordered tree directly
        # (reordered_tree is already a fresh copy we own)
        grafted_tree = reordered_tree.deep_copy()

        # CRITICAL: Always normalize ordering even without expand
        # The original algorithm always did this regardless of expand path length.
        # Skipping this causes "snapbacks" where leaf order jumps between frames.
        final_order = list(grafted_tree.get_current_order())

        # Reorder the reordered_tree copy that was added to the list
        if trees:
            trees[-1].reorder_taxa(final_order)

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
        subtree_tracker,
        expand_movers,
    )

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

    # Pre-compute sibling groups ONCE before processing any movers.
    # Phase-specific: collapse uses source parents, expand uses dest parents.
    collapse_splits = get_collapse_splits(collapse_paths_for_pivot_edge)
    expand_splits = get_expand_splits(expand_paths_for_pivot_edge)
    collapse_sibling_groups, expand_sibling_groups = compute_sibling_groups(
        all_mover_partitions,
        collapse_splits,
        expand_splits,
        source_parent_map,
        dest_parent_map,
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
                collapse_sibling_groups=collapse_sibling_groups,
                expand_sibling_groups=expand_sibling_groups,
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
