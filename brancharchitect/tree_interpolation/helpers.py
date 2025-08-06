"""
Helper and processing functions for tree interpolation.

This module contains utility functions for processing split data,
extracting data, and managing the interpolation workflow.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional
from brancharchitect.tree import Node
from brancharchitect.elements.partition import Partition
from brancharchitect.tree_interpolation.core import (
    calculate_intermediate,
    collapse_zero_length_branches_for_node,
    classical_interpolation,
)

logger: logging.Logger = logging.getLogger(__name__)


def _get_subset_splits(
    edge: Partition, current_splits: Dict[Partition, float]
) -> List[Partition]:
    """Find all splits that are subsets of a given edge."""
    return [split for split in current_splits if split.taxa.issubset(edge.taxa)]


def _filter_splits_by_subset(
    splits_dict: Dict[Partition, float],
    subset_splits: List[Partition],
) -> Dict[Partition, float]:
    """Filter a splits dictionary to only include specified subset splits.

    Args:
        splits_dict: The dictionary of splits with their weights
        subset_splits: List of splits to filter by

    Returns:
        A new dictionary containing only the splits that are in subset_splits
    """
    filtered_splits = {
        split: splits_dict[split] for split in subset_splits if split in splits_dict
    }
    return filtered_splits


def _reorder_consensus_tree_by_edge(
    consensus_tree: Node,
    target_tree: Node,
    edge: Partition,
) -> Node:
    """
    Create a reordered copy of the consensus tree based on target tree ordering
    by applying the reference node's order directly.

    This function leverages the robust `reorder_taxa` method from the Node
    class. It finds the corresponding node in the `target_tree`, gets its
    exact leaf order, and applies that order to the `consensus_tree` node.
    This correctly handles complex cases involving mixtures of leaf children
    and un-collapsed internal nodes (clades).

    Args:
        consensus_tree: The consensus tree to reorder.
        target_tree: The tree providing the target ordering.
        edge: The partition/edge identifying the node whose children to reorder.

    Returns:
        A new tree with the specified node reordered according to the target.
    """
    # Create a deep copy to avoid modifying the original
    reordered_tree = consensus_tree.deep_copy()

    # Find the corresponding nodes in both trees
    target_node = target_tree.find_node_by_split(edge)
    node_to_reorder = reordered_tree.find_node_by_split(edge)

    if target_node and node_to_reorder:
        # Get the definitive leaf order from the corresponding node in the target tree.
        correct_leaf_order = list(target_node.get_current_order())

        # Use the robust, built-in reorder_taxa method to apply the correct
        # order. This method is designed to handle complex sorting of both
        # leaves and internal clades.
        try:
            node_to_reorder.reorder_taxa(correct_leaf_order)
        except ValueError as e:
            logger.warning(
                f"Could not reorder node for s-edge {edge} due to taxa mismatch: {e}. "
                "The order may be partially inconsistent for this step."
            )

    return reordered_tree


def _create_down_phase_tree(
    base_tree: Node,
    s_edge: Partition,
    s_edge_subset_ref_weights: Dict[Partition, float],
    reference_weights: Dict[Partition, float],
) -> Node:
    """
    Create an intermediate tree for the down phase by applying reference weights.

    Args:
        base_tree: The tree to start from
        s_edge: The s-edge being processed
        s_edge_subset_ref_weights: Reference weights to apply within the s-edge
        reference_weights: Full reference weights for context

    Returns:
        A new tree with adjusted weights for the down phase, or unchanged tree if s-edge not found
    """
    intermediate_tree_down = base_tree.deep_copy()

    intermediate_edge_node = intermediate_tree_down.find_node_by_split(s_edge)

    if intermediate_edge_node is None:
        logger.warning(
            f"S-edge {s_edge} not found in down phase tree, skipping weight adjustment"
        )
        return intermediate_tree_down

    # Apply reference weights to s_edge branches, keep target weights elsewhere
    calculate_intermediate(
        intermediate_edge_node, s_edge_subset_ref_weights, reference_weights
    )

    return intermediate_tree_down


def _create_collapsed_consensus_tree(down_phase_tree: Node, s_edge: Partition) -> Node:
    collapsed_tree: Node = down_phase_tree.deep_copy()
    consensus_edge_node: Node | None = collapsed_tree.find_node_by_split(s_edge)
    if consensus_edge_node is not None:
        collapse_zero_length_branches_for_node(consensus_edge_node)
    root: Node = collapsed_tree.get_root()
    root.initialize_split_indices(root.taxa_encoding)
    root.invalidate_caches(propagate_up=True)
    return collapsed_tree


def _create_subtree_grafted_tree(
    reordered_tree: Node,
    reference_tree: Node,
    s_edge: Partition,
) -> Node:
    """
    Create a tree with the reference subtree's topology grafted at the s-edge position,
    preserving the order established in the reordering step.

    This function modifies a copy of the `reordered_tree` by updating the
    children of the `s_edge` node to match the `reference_tree`. This preserves
    the reordering of the `s_edge` node itself while updating its internal
    topology, ensuring order consistency between the reordered consensus tree
    and the pre-snap tree.

    Args:
        reordered_tree: The tree to graft into (already reordered).
        reference_tree: The tree to take the subtree topology from.
        s_edge: The s-edge identifying the node whose children to replace.

    Returns:
        A new tree with the reference subtree topology grafted in, or the
        reference subtree itself if replacing the root.
    """
    grafted_tree = reordered_tree.deep_copy()
    ref_node = reference_tree.find_node_by_split(s_edge)
    node = grafted_tree.find_node_by_split(s_edge)
    if node and ref_node and node.parent:
        node.children = [ch.deep_copy() for ch in ref_node.children]
        for ch in node.children:
            ch.parent = node
        root = grafted_tree.get_root()
        root.initialize_split_indices(root.taxa_encoding)
        root.invalidate_caches(propagate_up=True)
        return grafted_tree
    else:
        if ref_node:
            return ref_node.deep_copy()
        else:
            # Fallback if the reference subtree isn't found.
            return grafted_tree


def _create_pre_snap_tree(
    grafted_tree: Node,
    s_edge: Partition,
    s_edge_subset_target_weights: Dict[Partition, float],
    reference_weights: Dict[Partition, float],
) -> Node:
    """
    Create a pre-snap tree showing reference topology with target weights.

    Args:
        grafted_tree: The tree with reference topology (from grafting step)
        s_edge: The s-edge being processed
        s_edge_subset_target_weights: Target weights to apply within the s-edge
        reference_weights: Full reference weights for context

    Returns:
        A new tree with reference topology but target weights in the s-edge region, or unchanged tree if s-edge not found
    """
    pre_snap_tree = grafted_tree.deep_copy()
    edge_node = pre_snap_tree.find_node_by_split(s_edge)

    if edge_node is None:
        logger.warning(
            f"S-edge {s_edge} not found in pre-snap tree, skipping weight adjustment"
        )
        return pre_snap_tree

    # Apply target weights to show the contrast before full reference weights
    calculate_intermediate(edge_node, s_edge_subset_target_weights, reference_weights)

    return pre_snap_tree


def replace_weights_in_subset(
    current_weights: Dict[Partition, float],
    new_subset_weights: Dict[Partition, float],
    old_splits_to_delete: List[Partition],
) -> Dict[Partition, float]:
    """
    Replaces weights in a dictionary by deleting old splits and adding new ones.

    It first removes all splits specified in `old_splits_to_delete`
    and then adds all weights from `new_subset_weights`. This is not an
    in-place operation; it returns a new dictionary.

    Args:
        current_weights: The main dictionary of weights to be updated.
        new_subset_weights: A dictionary of new weights to add.
        old_splits_to_delete: A list of Partition keys to delete from
                                     current_weights.

    Returns:
        A new dictionary with the subset of weights replaced.
    """
    updated_weights = current_weights.copy()

    # Delete the old splits corresponding to the old topology
    for split in old_splits_to_delete:
        updated_weights.pop(split, None)  # Use pop with a default to avoid errors

    # Add the new splits corresponding to the new topology
    updated_weights.update(new_subset_weights)

    return updated_weights


def _create_classical_interpolation_fallback(
    current_state: Node,
    reference_tree: Node,
    reference_weights: Dict[Partition, float],
    s_edge: Partition,
    num_steps: int = 5,
) -> List[Node]:
    """
    Create a classical interpolation fallback when s-edge processing fails.

    Uses classical_interpolation to bridge from current state to reference tree.

    Args:
        current_state: The last successful interpolation state
        reference_tree: The target reference tree
        reference_weights: The reference tree weights
        s_edge: The s-edge that caused the failure (for logging)
        num_steps: Number of interpolation steps to generate

    Returns:
        A list of trees using classical interpolation
    """
    logger.info(f"Creating classical interpolation fallback for s-edge {s_edge}")

    try:
        # Calculate split data for classical interpolation
        current_splits = current_state.to_weighted_splits()
        split_data = (current_splits, reference_weights)

        # Use classical interpolation between current state and reference
        fallback_trees = classical_interpolation(
            current_state, reference_tree, split_data
        )
        # Ensure 5 trees are returned for consistency
        while len(fallback_trees) < num_steps:
            fallback_trees.append(fallback_trees[-1].deep_copy())
        return fallback_trees
    except Exception as e:
        logger.warning(f"Classical interpolation fallback failed: {e}")
        # Last resort: just return copies of current state
        return [current_state.deep_copy() for _ in range(num_steps)]


def generate_s_edge_interpolation_sequence(
    target_tree: Node,
    reference_tree: Node,
    reference_weights: Dict[Partition, float],
    target_s_edges: List[Partition],
    tree_index: int = 0,
) -> tuple[List[Node], List[Partition], List[str], List[Optional[Partition]]]:
    """
    Create an interpolation sequence from target to reference tree with integrated naming and tracking.

    For each s-edge, this function generates a sequence of five trees showing
    a detailed transition that interpolates from the target tree's topology
    toward the reference tree's topology. Tree names and s-edge tracking are
    generated step by step alongside the interpolation sequence.

    Args:
        target_tree: The starting tree for interpolation.
        reference_tree: The destination tree for topology and ordering.
        reference_weights: Branch weights from the reference tree.
        target_s_edges: S-edges to process in order.
        tree_index: Index for tree naming (default: 0).

    Returns:
        A tuple containing:
        - A list of trees showing the interpolation sequence from target to reference
        - A list of s-edges that failed and used classical interpolation fallback
        - A list of names corresponding to each tree in the interpolation sequence
        - A list of s-edges being processed for each tree (None for classical interpolation)
    """
    if not target_s_edges:
        logger.info("target_s_edges is empty, returning an empty list of trees.")
        return [], [], [], []

    interpolation_sequence: List[Node] = []
    failed_s_edges: List[Partition] = []
    tree_names: List[str] = []
    s_edge_tracking: List[Optional[Partition]] = []
    failed_set: set[Partition] = set()  # Will be updated as we find failures
    logger.info(f"Creating interpolation sequence for {len(target_s_edges)} s-edges.")
    interpolation_state: Node = target_tree.deep_copy()

    for i, s_edge in enumerate(target_s_edges):
        current_base_tree = interpolation_state.deep_copy()
        current_base_tree.initialize_split_indices(current_base_tree.taxa_encoding)
        s_edge_in_current_state = current_base_tree.find_node_by_split(s_edge)
        s_edge_in_reference = reference_tree.find_node_by_split(s_edge)

        if s_edge_in_current_state and s_edge_in_reference:
            #
            # For each s_edge, we create a 5-step interpolation subsequence:
            # Step 1: Weight adjustment - blend reference weights into target topology
            # Step 2: Collapse - remove zero-length branches to simplify
            # Step 3: Reorder - match reference tree's node ordering
            # Step 4: Pre-snap - show reference topology with target weights (contrast)
            # Step 5: Snap - apply full reference weights to complete the transition
            #
            # NOTE: Steps 4 and 5 are computed out of order (5 then 4) because
            # step 4 needs the reference topology created in step 5, but they are
            # appended to the sequence in the correct visual order (4 then 5)

            # Step 1: Weight adjustment - Apply reference weights to s_edge subset while keeping other target weights
            # Extract reference weights for only the branches within this s_edge

            reference_subset_splits: List[Partition] = _get_subset_splits(
                s_edge, reference_weights
            )

            s_edge_subset_ref_weights: Dict[Partition, float] = (
                _filter_splits_by_subset(reference_weights, reference_subset_splits)
            )

            intermediate_tree_down: Node = _create_down_phase_tree(
                current_base_tree, s_edge, s_edge_subset_ref_weights, reference_weights
            )

            interpolation_sequence.append(intermediate_tree_down)
            tree_names.append(f"IT{tree_index}_down_{i + 1}")
            s_edge_tracking.append(s_edge)

            # Step 2: Collapse - Remove zero-length branches to create consensus
            # This simplifies the topology by removing branches that have been zeroed out
            interp_step2_collapsed: Node = _create_collapsed_consensus_tree(
                intermediate_tree_down, s_edge
            )
            interpolation_sequence.append(interp_step2_collapsed)
            tree_names.append(f"C{tree_index}_{i + 1}")
            s_edge_tracking.append(s_edge)

            # Step 3: Reorder - Match reference tree's node ordering
            # This prepares for the snap by ensuring child nodes are in the same order
            interp_step3_reordered_to_ref: Node = _reorder_consensus_tree_by_edge(
                interp_step2_collapsed, reference_tree, s_edge
            )
            interpolation_sequence.append(interp_step3_reordered_to_ref)
            tree_names.append(f"C{tree_index}_{i + 1}_reorder")
            s_edge_tracking.append(s_edge)

            # Step 5: Snap to reference - Replace s_edge subtree with reference subtree
            # This is the key topology change: we swap in the reference tree's structure
            interpolation_state = _create_subtree_grafted_tree(
                interp_step3_reordered_to_ref, reference_tree, s_edge
            )

            # Step 4: Pre-snap preparation - Adjust weights before final snap
            # This step shows the new reference topology but with the old target weights
            # applied only to the subset of branches being changed in this step.

            # Get weights from the current state of the tree *before* this snap.
            current_weights = current_base_tree.to_weighted_splits()

            # Filter to get the weights for the specific subset of branches being transformed.
            s_edge_subset_target_weights = _filter_splits_by_subset(
                current_weights, reference_subset_splits
            )

            interp_step4_pre_snap = _create_pre_snap_tree(
                interpolation_state,
                s_edge,
                s_edge_subset_target_weights,  # Use the filtered target weights
                reference_weights,
            )

            interpolation_sequence.append(interp_step4_pre_snap)
            tree_names.append(f"IT{tree_index}_up_{i + 1}")
            s_edge_tracking.append(s_edge)
            interpolation_sequence.append(interpolation_state)  # This is the 5th tree
            tree_names.append(f"IT{tree_index}_ref_{i + 1}")
            s_edge_tracking.append(s_edge)

            logger.debug(
                f"Step {i + 1}/{len(target_s_edges)}: Generated 5 trees for s-edge {s_edge}"
            )
            logger.info(
                f"DEBUG: generate_s_edge_interpolation_sequence - After s_edge {s_edge}: len(interpolation_sequence)={len(interpolation_sequence)}"
            )

        else:
            logger.warning(
                f"s-edge {s_edge} not found in both trees. Using classical interpolation fallback."
            )
            # Track this s-edge as failed
            failed_s_edges.append(s_edge)
            failed_set.add(s_edge)  # Update failed set for naming
            # Fallback: Use classical interpolation between current state and reference
            fallback_trees = _create_classical_interpolation_fallback(
                interpolation_state, reference_tree, reference_weights, s_edge
            )
            interpolation_sequence.extend(fallback_trees)

            # CRITICAL FIX: Update the state to the last tree from the fallback sequence
            if fallback_trees:
                interpolation_state = fallback_trees[-1].deep_copy()

            # Add classical interpolation names
            tree_names.extend(
                [
                    f"IT{tree_index}_classical_{i + 1}_1",
                    f"IT{tree_index}_classical_{i + 1}_2",
                    f"IT{tree_index}_classical_{i + 1}_3",
                    f"IT{tree_index}_classical_{i + 1}_4",
                    f"IT{tree_index}_classical_{i + 1}_5",
                ]
            )
            # Classical interpolation doesn't use specific s-edge
            s_edge_tracking.extend([None] * 5)

    return interpolation_sequence, failed_s_edges, tree_names, s_edge_tracking
