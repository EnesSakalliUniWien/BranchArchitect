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
from brancharchitect.tree_interpolation.consensus_tree.consensus_tree import (
    reorder_consensus_tree_by_edge,
    create_collapsed_consensus_tree,
)
from brancharchitect.tree_interpolation.consensus_tree.intermediate_tree import (
    create_down_phase_tree,
    create_pre_snap_tree,
)
from brancharchitect.tree_interpolation.helpers import (
    calculate_subtree_paths,
    get_subset_splits,
    filter_splits_by_subset,
    create_subtree_grafted_tree,
    create_classical_interpolation_fallback,
)


logger: logging.Logger = logging.getLogger(__name__)


def generate_s_edge_interpolation_sequence(
    target_tree: Node,
    reference_tree: Node,
    reference_weights: Dict[Partition, float],
    target_s_edges: List[Partition],
    s_edge_solutions: Dict[Partition, List[List[Partition]]],
    tree_index: int = 0,
    mapping_one: Optional[Dict[Partition, Partition]] = None,
    mapping_two: Optional[Dict[Partition, Partition]] = None,
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
    interpolation_sequence: List[Node] = []
    failed_s_edges: List[Partition] = []
    tree_names: List[str] = []
    s_edge_tracking: List[Optional[Partition]] = []
    failed_set: set[Partition] = set()  # Will be updated as we find failures
    logger.info(f"Creating interpolation sequence for {len(target_s_edges)} s-edges.")
    interpolation_state: Node = target_tree.deep_copy()

    reference_subtree_paths, target_subtree_paths = calculate_subtree_paths(
        s_edge_solutions, reference_tree, target_tree
    )

    # Debug prints removed for cleaner logs

    for i, s_edge in enumerate(target_s_edges):
        current_base_tree: Node = interpolation_state.deep_copy()
        current_base_tree.initialize_split_indices(current_base_tree.taxa_encoding)
        s_edge_in_current_state: Node | None = current_base_tree.find_node_by_split(
            s_edge
        )
        s_edge_in_reference: Node | None = reference_tree.find_node_by_split(s_edge)

        if s_edge_in_current_state and s_edge_in_reference:
            # For each s_edge, we create a 5-step interpolation subsequence:
            # Step 1: Weight adjustment - blend reference weights into target topology
            # Step 2: Collapse - remove zero-length branches to simplify
            # Step 3: Reorder - match reference tree's node ordering
            # Step 4: Pre-snap - show reference topology with target weights (contrast)
            # Step 5: Snap - apply full reference weights to complete the transition
            # NOTE: Steps 4 and 5 are computed out of order (5 then 4) because
            # step 4 needs the reference topology created in step 5, but they are
            # appended to the sequence in the correct visual order (4 then 5)
            # Step 1: Weight adjustment - Apply reference weights to s_edge subset while keeping other target weights
            # Extract reference weights for only the branches within this s_edge
            reference_subset_splits: List[Partition] = get_subset_splits(
                s_edge, reference_weights
            )

            s_edge_subset_ref_weights: Dict[Partition, float] = filter_splits_by_subset(
                reference_weights, reference_subset_splits
            )

            intermediate_tree_down: Node = create_down_phase_tree(
                current_base_tree,
                s_edge,
                s_edge_subset_ref_weights,
                reference_weights,
            )

            interpolation_sequence.append(intermediate_tree_down)
            tree_names.append(f"IT{tree_index}_down_{i + 1}")
            s_edge_tracking.append(s_edge)
            # Step 2: Collapse - Remove zero-length branches to create consensus
            # This simplifies the topology by removing branches that have been zeroed out
            interp_step2_collapsed: Node = create_collapsed_consensus_tree(
                intermediate_tree_down, s_edge
            )
            interpolation_sequence.append(interp_step2_collapsed)
            tree_names.append(f"C{tree_index}_{i + 1}")
            s_edge_tracking.append(s_edge)
            # Step 3: Reorder - Match reference tree's node ordering
            # This prepares for the snap by ensuring child nodes are in the same order
            interp_step3_reordered_to_ref: Node = reorder_consensus_tree_by_edge(
                interp_step2_collapsed, reference_tree, s_edge
            )

            interpolation_sequence.append(interp_step3_reordered_to_ref)
            tree_names.append(f"C{tree_index}_{i + 1}_reorder")
            s_edge_tracking.append(s_edge)
            # Step 5: Snap to reference - Replace s_edge subtree with reference subtree
            # This is the key topology change: we swap in the reference tree's structure
            interpolation_state = create_subtree_grafted_tree(
                interp_step3_reordered_to_ref, reference_tree, s_edge
            )
            # Step 4: Pre-snap preparation - Adjust weights before final snap
            # This step shows the new reference topology but with the old target weights
            # applied only to the subset of branches being changed in this step.
            # Get weights from the current state of the tree *before* this snap.
            current_weights: Dict[Partition, float] = (
                current_base_tree.to_weighted_splits()
            )
            # Filter to get the weights for the specific subset of branches being transformed.
            s_edge_subset_target_weights: Dict[Partition, float] = (
                filter_splits_by_subset(current_weights, reference_subset_splits)
            )

            interp_step4_pre_snap: Node = create_pre_snap_tree(
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
            fallback_trees = create_classical_interpolation_fallback(
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
