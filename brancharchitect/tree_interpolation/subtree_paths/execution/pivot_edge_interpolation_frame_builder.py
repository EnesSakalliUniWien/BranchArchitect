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
from brancharchitect.tree_interpolation.types import SprMoveEvent, SprPathSegment
from ..planning import build_edge_plan
from .reordering import reorder_tree_toward_destination, align_to_source_order
from .sibling_grouping import (
    compute_sibling_groups,
    get_collapse_splits,
    get_expand_splits,
    get_group_for_mover,
)

logger = logging.getLogger(__name__)
_MISSING_VISUAL_ORDER = 10**12


# ============================================================================
# Microstep Building Functions
# ============================================================================


def _append_frame(
    trees: List[Node],
    edges: List[Optional[Partition]],
    tree: Node,
    edge: Optional[Partition],
    subtree_highlight_tracker: List[List[Partition]],
    highlight_group: List[Partition],
) -> None:
    """Append an animation frame to the output lists.

    Args:
        trees: List to append the tree to
        edges: List to append the edge to
        tree: The tree to add
        edge: The edge to add
        subtree_highlight_tracker: Per-frame visual/highlight groups
        highlight_group: Partitions visually associated with this frame
    """
    trees.append(tree)
    edges.append(edge)
    subtree_highlight_tracker.append(list(highlight_group))


def build_frames_for_subtree(
    interpolation_state: Node,
    destination_tree: Node,
    current_pivot_edge: Partition,
    selection: Dict[str, Any],
    all_mover_partitions: Optional[List[Partition]] = None,
    source_parent_map: Optional[Dict[Partition, Partition]] = None,
    dest_parent_map: Optional[Dict[Partition, Partition]] = None,
    is_first_mover: bool = True,
    source_weights: Optional[Dict[Partition, float]] = None,
    destination_weights: Optional[Dict[Partition, float]] = None,
    collapse_sibling_groups: Optional[Dict[Partition, List[Partition]]] = None,
    expand_sibling_groups: Optional[Dict[Partition, List[Partition]]] = None,
) -> Tuple[List[Node], List[Optional[Partition]], Node, List[List[Partition]]]:
    """
    Build animation frames for one planner-selected driver under an active-changing edge.

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
        all_mover_partitions: List of all mover Partitions (blocks) for this pivot edge.
        source_parent_map: Maps each mover -> its parent in source tree.
        dest_parent_map: Maps each mover -> its parent in destination tree.
        is_first_mover: Whether this is the first planner-selected mover for the current pivot edge.
        source_weights: Optional dictionary of original source weights for stable averaging.

    Returns:
        Tuple containing:
        - List[Node]: Animation frames (intermediate trees) for this selection.
        - List[Optional[Partition]]: The pivot edge associated with each frame.
        - Node: The final tree state after all transformations (always a fresh copy).
        - List[List[Partition]]: Visual/highlight groups associated with each frame.
    """
    trees: List[Node] = []
    edges: List[Optional[Partition]] = []
    subtree_highlight_tracker: List[List[Partition]] = []

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
    # Determine phase-specific visual/highlight groups.
    # These groups provide context for a frame; they are not an assertion that
    # every listed partition physically moves in that microstep.
    # =========================================================================
    # Collapse phase: use source parent grouping
    collapse_highlight_group: List[Partition] = [subtree_partition]
    if collapse_sibling_groups:
        collapse_highlight_group = get_group_for_mover(
            subtree_partition, collapse_sibling_groups
        )

    # Expand phase: use dest parent grouping
    expand_highlight_group: List[Partition] = [subtree_partition]
    if expand_sibling_groups:
        expand_highlight_group = get_group_for_mover(
            subtree_partition, expand_sibling_groups
        )

    reorder_highlight_group: List[Partition] = []
    seen_reorder_highlights: set[Partition] = set()
    for sibling_groups in (collapse_sibling_groups, expand_sibling_groups):
        if sibling_groups:
            for mover in get_group_for_mover(subtree_partition, sibling_groups):
                if mover not in seen_reorder_highlights:
                    reorder_highlight_group.append(mover)
                    seen_reorder_highlights.add(mover)
    if not reorder_highlight_group:
        reorder_highlight_group = [subtree_partition]

    pending_frame: Optional[Tuple[Node, Optional[Partition], List[Partition]]] = None

    def flush_pending_frame_snapshot(final_order: Optional[List[str]] = None) -> None:
        nonlocal pending_frame
        if pending_frame is None:
            return

        tree, edge, highlight_group = pending_frame
        if final_order is not None:
            tree.reorder_taxa(final_order)
        _append_frame(
            trees,
            edges,
            tree,
            edge,
            subtree_highlight_tracker,
            highlight_group,
        )
        pending_frame = None

    def set_pending_frame_snapshot(
        tree: Node, edge: Optional[Partition], highlight_group: List[Partition]
    ) -> None:
        """Take ownership of a tree that will become an immutable output frame."""
        nonlocal pending_frame
        flush_pending_frame_snapshot()
        pending_frame = (tree, edge, list(highlight_group))

    # =========================================================================
    # Phase 1: Collapse (compute always, add frames conditionally)
    # =========================================================================
    if has_collapse_work:
        zeroed_tree: Node = interpolation_state.deep_copy(build_split_index=False)

        apply_zero_branch_lengths(zeroed_tree, PartitionSet(set(collapse_paths)))

        collapsed_tree: Node = create_collapsed_consensus_tree(
            zeroed_tree,
            current_pivot_edge,
            copy=True,
            destination_tree=destination_tree,
        )

        set_pending_frame_snapshot(
            zeroed_tree,
            current_pivot_edge,
            collapse_highlight_group,
        )

        set_pending_frame_snapshot(
            collapsed_tree.deep_copy(build_split_index=False),
            current_pivot_edge,
            collapse_highlight_group,
        )
        collapsed_tree_owned = True

    else:
        # No collapse work: borrow current state until a frame or mutation needs
        # its own tree.
        collapsed_tree = interpolation_state
        collapsed_tree_owned = False

    # =========================================================================
    # Phase 2: Reorder (compute always, add frames conditionally)
    # =========================================================================
    pre_reorder_order = tuple(collapsed_tree.get_current_order())
    reordered_tree: Node = reorder_tree_toward_destination(
        source_tree=collapsed_tree,
        destination_tree=destination_tree,
        current_pivot_edge=current_pivot_edge,
        moving_subtree_partition=subtree_partition,
        source_parent_map=source_parent_map,
        dest_parent_map=dest_parent_map,
        # Context only: these movers are unstable non-anchors, but only
        # subtree_partition moves during this microstep.
        unstable_mover_partitions=all_mover_partitions,
        copy=not collapsed_tree_owned,
    )

    has_reorder_change = tuple(reordered_tree.get_current_order()) != pre_reorder_order
    reordered_tree_owned = has_reorder_change or collapsed_tree_owned

    if has_reorder_change:
        # Add reorder frames: before and after
        if not has_collapse_work:
            # Need to show pre-reorder state (collapsed_tree wasn't added yet)

            pre_reorder_frame = (
                collapsed_tree
                if collapsed_tree_owned
                else collapsed_tree.deep_copy(build_split_index=False)
            )
            set_pending_frame_snapshot(
                pre_reorder_frame,
                current_pivot_edge,
                reorder_highlight_group,
            )

        reorder_frame_tree = (
            reordered_tree.deep_copy(build_split_index=False)
            if not is_first_mover and not has_expand_work
            else reordered_tree
        )
        set_pending_frame_snapshot(
            reorder_frame_tree,
            current_pivot_edge,
            reorder_highlight_group,
        )
    else:
        # No reorder change: continue with the same tree. In no-collapse cases
        # this may still be a borrowed state.
        reordered_tree = collapsed_tree

    # =========================================================================
    # Early exit if no further work
    # =========================================================================
    if not is_first_mover and not has_expand_work:
        # No expand or snap work - return reordered state
        flush_pending_frame_snapshot()
        if not trees:
            # No frames at all - still return a copy
            if not reordered_tree_owned:
                reordered_tree = reordered_tree.deep_copy()
            return [], [], reordered_tree, []
        return trees, edges, reordered_tree, subtree_highlight_tracker

    # =========================================================================
    # Phase 3: Expand/Graft (compute and add frames conditionally)
    # =========================================================================
    if destination_weights is None:
        destination_weights = destination_tree.to_weighted_splits()

    if has_expand_work:
        reordered_order = list(reordered_tree.get_current_order())

        if pending_frame is not None and pending_frame[0] is reordered_tree:
            pending_frame = (
                reordered_tree.deep_copy(build_split_index=False),
                pending_frame[1],
                pending_frame[2],
            )

        if not reordered_tree_owned:
            reordered_tree = reordered_tree.deep_copy()
            reordered_tree_owned = True

        grafted_zero_weights: Node = create_subtree_grafted_tree(
            base_tree=reordered_tree,
            ref_path_to_build=expand_paths,
            copy=False,
        )

        grafted_zero_weights.reorder_taxa(reordered_order)

        # Use align_to_source_order to preserve non-mover positions after graft
        align_to_source_order(
            grafted_zero_weights,
            source_order=reordered_order,
            moving_taxa=_taxa_for_partitions(expand_highlight_group),
        )

        # Capture final order BEFORE applying weights (more defensive)
        # Grafting may introduce new ordering that collapsed consensus doesn't have
        final_order = list(grafted_zero_weights.get_current_order())

        # Normalize the pending pre-expand frame before it is emitted. This keeps
        # frame construction local: once appended, a frame is never rewritten.
        flush_pending_frame_snapshot(final_order)

        # Add expand frame (grafted with zero weights) before mutating the owned
        # grafted tree into the weighted snap/final state.
        _append_frame(
            trees,
            edges,
            grafted_zero_weights.deep_copy(build_split_index=False),
            current_pivot_edge,
            subtree_highlight_tracker,
            expand_highlight_group,
        )

        grafted_tree = grafted_zero_weights

        # Apply weights to snap tree (doesn't change ordering)
        finalize_branch_weights(
            tree=grafted_tree,
            current_pivot_edge=current_pivot_edge,
            expand_path=expand_paths,
            is_first_mover=is_first_mover,
            source_weights=source_weights,
            destination_weights=destination_weights,
        )

        snap_highlight_group = expand_highlight_group

    else:
        # No expand work - snap operates on reordered tree directly
        # (reordered_tree is already a fresh copy we own)
        grafted_tree = reordered_tree.deep_copy()

        # CRITICAL: Always normalize ordering even without expand
        # The original algorithm always did this regardless of expand path length.
        # Skipping this causes "snapbacks" where leaf order jumps between frames.
        final_order = list(grafted_tree.get_current_order())

        # Normalize the pending pre-snap frame before it is emitted. This keeps
        # frame construction local: once appended, a frame is never rewritten.
        flush_pending_frame_snapshot(final_order)

        # Apply weights to snap tree
        finalize_branch_weights(
            tree=grafted_tree,
            current_pivot_edge=current_pivot_edge,
            expand_path=expand_paths,
            is_first_mover=is_first_mover,
            source_weights=source_weights,
            destination_weights=destination_weights,
        )
        snap_highlight_group = reorder_highlight_group

    # =========================================================================
    # Phase 4: Snap frame
    # =========================================================================
    _append_frame(
        trees,
        edges,
        grafted_tree.deep_copy(build_split_index=False),
        current_pivot_edge,
        subtree_highlight_tracker,
        snap_highlight_group,
    )

    return trees, edges, grafted_tree, subtree_highlight_tracker


def _taxa_for_partitions(partitions: List[Partition]) -> set[str]:
    taxa: set[str] = set()
    for partition in partitions:
        taxa.update(partition.taxa)
    return taxa


# ============================================================================
# Edge Plan Execution
# ============================================================================


def _path_segments_with_branch_lengths(
    path: List[Partition],
    weights: Dict[Partition, float],
    side: str,
) -> List[SprPathSegment]:
    segments: List[SprPathSegment] = []
    for split in path:
        if split not in weights:
            raise KeyError(
                f"Missing {side} branch length for SPR path split {split.indices}"
            )
        segments.append(
            {
                "split": split,
                "branch_length": float(weights[split]),
            }
        )
    return segments


def _build_spr_move_event(
    current_pivot_edge: Partition,
    driver_subtree: Partition,
    highlight_group: List[Partition],
    selection: Dict[str, Any],
    source_weights: Dict[Partition, float],
    destination_weights: Dict[Partition, float],
    step_range: Tuple[int, int],
) -> SprMoveEvent:
    collapse_path: List[Partition] = selection.get("collapse", {}).get(
        "path_segment", []
    )
    expand_path: List[Partition] = selection.get("expand", {}).get("path_segment", [])

    collapse_segments = _path_segments_with_branch_lengths(
        collapse_path, source_weights, "source"
    )
    expand_segments = _path_segments_with_branch_lengths(
        expand_path, destination_weights, "destination"
    )
    collapse_branch_length = sum(
        segment["branch_length"] for segment in collapse_segments
    )
    expand_branch_length = sum(segment["branch_length"] for segment in expand_segments)
    collapse_hops = len(collapse_segments)
    expand_hops = len(expand_segments)

    return {
        "pivot_edge": current_pivot_edge,
        "driver_subtree": driver_subtree,
        "highlight_group": highlight_group,
        "step_range": step_range,
        "collapse_path": collapse_segments,
        "expand_path": expand_segments,
        "collapse_hops": collapse_hops,
        "expand_hops": expand_hops,
        "total_hops": collapse_hops + expand_hops,
        "collapse_branch_length": collapse_branch_length,
        "expand_branch_length": expand_branch_length,
        "total_branch_length": collapse_branch_length + expand_branch_length,
    }


def _highlight_group_for_event(
    subtree: Partition, step_highlight_tracker: List[List[Partition]]
) -> List[Partition]:
    """Build the public SPR visual group with the planner driver included once."""
    group: List[Partition] = [subtree]
    seen: set[Partition] = {subtree}
    for frame_group in step_highlight_tracker:
        for mover in frame_group:
            if mover not in seen:
                group.append(mover)
                seen.add(mover)
    return group


def _build_destination_mover_order_key(
    destination_tree: Node,
    current_pivot_edge: Partition,
    mover_partitions: set[Partition],
) -> Dict[Partition, Tuple[int, ...]]:
    destination_subtree = destination_tree.find_node_by_split(current_pivot_edge)
    destination_order = (
        destination_subtree.get_current_order()
        if destination_subtree is not None
        else destination_tree.get_current_order()
    )
    position_by_taxon = {taxon: index for index, taxon in enumerate(destination_order)}
    fallback_position = len(position_by_taxon)

    order_key: Dict[Partition, Tuple[int, ...]] = {}
    for mover in mover_partitions:
        positions = sorted(
            position_by_taxon[taxon]
            for taxon in mover.taxa
            if taxon in position_by_taxon
        )
        if not positions:
            order_key[mover] = (
                fallback_position,
                fallback_position,
                fallback_position,
                0,
            )
            continue

        order_key[mover] = (
            positions[0],
            positions[-1],
            sum(positions),
            len(positions),
        )

    return order_key


def execute_pivot_edge_plan(
    current_base_tree: Node,
    destination_tree: Node,
    source_tree: Node,
    current_pivot_edge: Partition,
    expand_paths_for_pivot_edge: Dict[Partition, PartitionSet[Partition]],
    collapse_paths_for_pivot_edge: Dict[Partition, PartitionSet[Partition]],
    source_parent_map: Optional[Dict[Partition, Partition]] = None,
    dest_parent_map: Optional[Dict[Partition, Partition]] = None,
) -> Tuple[
    List[Node],
    List[Optional[Partition]],
    Node,
    List[List[Partition]],
    List[SprMoveEvent],
]:
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
        Tuple of (trees, edges, interpolation_state, subtree_highlight_tracker)
    """
    trees: List[Node] = []
    edges: List[Optional[Partition]] = []
    subtree_highlight_tracker: List[List[Partition]] = []
    spr_move_events: List[SprMoveEvent] = []
    interpolation_state: Node = current_base_tree
    expand_paths_for_plan = dict(expand_paths_for_pivot_edge)
    collapse_paths_for_plan = dict(collapse_paths_for_pivot_edge)
    initial_mover_partition_set = set(expand_paths_for_plan) | set(
        collapse_paths_for_plan
    )
    initial_subtree_order_key = _build_destination_mover_order_key(
        destination_tree,
        current_pivot_edge,
        initial_mover_partition_set,
    )

    selections: Dict[Partition, Dict[str, Any]] = build_edge_plan(
        expand_paths_for_plan,
        collapse_paths_for_plan,
        source_tree,  # Use original source tree for split computation, NOT interpolation state
        destination_tree,
        current_pivot_edge=current_pivot_edge,
        subtree_order_key=initial_subtree_order_key,
    )

    # Calculate source weights once using the ORIGINAL source tree
    # This ensures consistent (Source + Dest) / 2 interpolation
    source_weights: Dict[Partition, float] = source_tree.to_weighted_splits()
    destination_weights: Dict[Partition, float] = destination_tree.to_weighted_splits()

    # All mover partitions as BLOCKS (not flattened to taxa)
    # CRITICAL: We must include ALL subtrees that have paths, even if they were dropped
    # from the plan (Passenger subtrees handled by Drivers).
    # Using selections.keys() would miss these passengers, causing split grouping logic to fail.
    mover_partition_set = (
        set(expand_paths_for_plan) | set(collapse_paths_for_plan) | set(selections)
    )
    subtree_order_key = _build_destination_mover_order_key(
        destination_tree,
        current_pivot_edge,
        mover_partition_set,
    )
    all_mover_partitions: List[Partition] = sorted(
        mover_partition_set,
        key=lambda p: (*subtree_order_key.get(p, (_MISSING_VISUAL_ORDER,)), p.bitmask),
    )

    # Pre-compute sibling groups ONCE before processing any movers.
    # Phase-specific: collapse uses source parents, expand uses dest parents.
    collapse_splits = get_collapse_splits(collapse_paths_for_plan)
    expand_splits = get_expand_splits(expand_paths_for_plan)
    collapse_sibling_groups, expand_sibling_groups = compute_sibling_groups(
        all_mover_partitions,
        collapse_splits,
        expand_splits,
        source_parent_map,
        dest_parent_map,
    )

    # Preserve planner order while tracking the first mover for weight averaging.
    selection_items = list(selections.items())

    for i, (subtree, selection) in enumerate(selection_items):
        is_first_mover = i == 0

        # Add subtree to selection for compatibility
        selection_with_subtree: Dict[str, Any] = {**selection, "subtree": subtree}

        step_start = len(trees)
        step_trees, step_edges, interpolation_state, step_highlight_tracker = (
            build_frames_for_subtree(
                interpolation_state=interpolation_state,
                destination_tree=destination_tree,
                current_pivot_edge=current_pivot_edge,
                selection=selection_with_subtree,
                all_mover_partitions=all_mover_partitions,
                source_parent_map=source_parent_map,
                dest_parent_map=dest_parent_map,
                is_first_mover=is_first_mover,
                source_weights=source_weights,
                destination_weights=destination_weights,
                collapse_sibling_groups=collapse_sibling_groups,
                expand_sibling_groups=expand_sibling_groups,
            )
        )
        if step_trees:
            step_end = step_start + len(step_trees) - 1
            spr_move_events.append(
                _build_spr_move_event(
                    current_pivot_edge=current_pivot_edge,
                    driver_subtree=subtree,
                    highlight_group=_highlight_group_for_event(
                        subtree, step_highlight_tracker
                    ),
                    selection=selection_with_subtree,
                    source_weights=source_weights,
                    destination_weights=destination_weights,
                    step_range=(step_start, step_end),
                )
            )

        trees.extend(step_trees)
        edges.extend(step_edges)
        subtree_highlight_tracker.extend(step_highlight_tracker)

    # Handle edge case: no selections means no work to do for this pivot edge
    if not trees:
        # Add the current state as a pass-through step so interpolation can continue.
        trees.append(interpolation_state.deep_copy())
        edges.append(current_pivot_edge)
        subtree_highlight_tracker.append([])

    return trees, edges, interpolation_state, subtree_highlight_tracker, spr_move_events
