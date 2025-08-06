"""
Tree interpolation module for creating smooth animations between phylogenetic trees.

This module provides the main public API for tree interpolation, creating
intermediate states that allow continuous morphing from one tree topology to another.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Dict

from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree_interpolation.core import (
    calculate_consensus_tree,
    calculate_intermediate_tree,
)
from brancharchitect.tree import Node
from brancharchitect.tree_interpolation.types import (
    TreePairInterpolation,
    LatticeEdgeData,
    TreeInterpolationSequence,
)
from brancharchitect.tree_interpolation.helpers import (
    generate_s_edge_interpolation_sequence,
)

# Import here to avoid circular import
from brancharchitect.jumping_taxa.lattice.lattice_solver import (
    iterate_lattice_algorithm,
)

# Generate solution to atom mappings
from brancharchitect.jumping_taxa.lattice.mapping import (
    map_solutions_to_atoms,
)


logger: logging.Logger = logging.getLogger(__name__)

__all__: List[str] = [
    "interpolate_tree",
    "interpolate_adjacent_tree_pairs",
    "build_sequential_lattice_interpolations",
]


def _calculate_s_edge_distances(
    target: Node,
    reference: Node,
    lattice_edge_solutions: Dict[Partition, List[List[Partition]]],
) -> Dict[Partition, Dict[str, float]]:
    """
    Calculate distances from jumping taxa components to their corresponding s-edges.

    For each s-edge in the lattice solutions, computes both topological (unweighted)
    and branch length weighted distances from all jumping taxa (components) to the
    s-edge node in both target and reference trees.

    Args:
        target: Target tree for interpolation
        reference: Reference tree for interpolation
        lattice_edge_solutions: Dictionary mapping s-edges to their solution sets

    Returns:
        Dictionary mapping each s-edge to distance metrics:
        - "target_topological": Average topological distance in target tree (edge count)
        - "target_weighted": Average branch length weighted distance in target tree
        - "reference_topological": Average topological distance in reference tree (edge count)
        - "reference_weighted": Average branch length weighted distance in reference tree
        - "total_topological": Sum of target and reference topological distances
        - "total_weighted": Sum of target and reference weighted distances
        - "component_count": Number of jumping taxa for this s-edge
    """
    s_edge_distances: Dict[Partition, Dict[str, float]] = {}

    for s_edge, solution_sets in lattice_edge_solutions.items():
        target_topological_distances: List[float] = []
        target_weighted_distances: List[float] = []
        reference_topological_distances: List[float] = []
        reference_weighted_distances: List[float] = []
        total_components = 0

        # Process all solution sets for this s-edge
        for solution_set in solution_sets:
            for component in solution_set:
                total_components += 1

                # Calculate distances from component to s-edge in target tree
                target_path = target.find_path_between_splits(component, s_edge)
                if target_path:
                    # Topological distance: number of edges in path
                    target_topo_dist = (
                        len(target_path) - 1
                    )  # -1 because path includes both endpoints
                    target_topological_distances.append(target_topo_dist)

                    # Weighted distance: sum of branch lengths in path
                    target_weighted_dist = sum(
                        node.length if node.length is not None else 0.0
                        for node in target_path[
                            1:
                        ]  # Skip first node (component node itself)
                    )
                    target_weighted_distances.append(target_weighted_dist)
                else:
                    target_topological_distances.append(0.0)
                    target_weighted_distances.append(0.0)

                # Calculate distances from component to s-edge in reference tree
                reference_path = reference.find_path_between_splits(component, s_edge)
                if reference_path:
                    # Topological distance: number of edges in path
                    reference_topo_dist = len(reference_path) - 1
                    reference_topological_distances.append(reference_topo_dist)

                    # Weighted distance: sum of branch lengths in path
                    reference_weighted_dist = sum(
                        node.length if node.length is not None else 0.0
                        for node in reference_path[
                            1:
                        ]  # Skip first node (component node itself)
                    )
                    reference_weighted_distances.append(reference_weighted_dist)
                else:
                    reference_topological_distances.append(0.0)
                    reference_weighted_distances.append(0.0)

        # Calculate average distances
        if total_components > 0:
            avg_target_topo = sum(target_topological_distances) / len(
                target_topological_distances
            )
            avg_target_weighted = sum(target_weighted_distances) / len(
                target_weighted_distances
            )
            avg_reference_topo = sum(reference_topological_distances) / len(
                reference_topological_distances
            )
            avg_reference_weighted = sum(reference_weighted_distances) / len(
                reference_weighted_distances
            )
        else:
            avg_target_topo = 0.0
            avg_target_weighted = 0.0
            avg_reference_topo = 0.0
            avg_reference_weighted = 0.0

        s_edge_distances[s_edge] = {
            "target_topological": avg_target_topo,
            "target_weighted": avg_target_weighted,
            "reference_topological": avg_reference_topo,
            "reference_weighted": avg_reference_weighted,
            "total_topological": avg_target_topo + avg_reference_topo,
            "total_weighted": avg_target_weighted + avg_reference_weighted,
            "component_count": float(total_components),
        }

    return s_edge_distances


# Public API
def interpolate_tree(target: Node, reference: Node) -> tuple[Node, Node, Node, Node]:
    """
    Interpolate between two trees to create intermediate and consensus trees.

    Returns a tuple of 4 trees:
    1. Intermediate tree from target (branch lengths averaged toward reference)
    2. Consensus from target (keeping only splits that are also in reference)
    3. Consensus from reference (keeping only splits that are also in target)
    4. Intermediate tree from reference (branch lengths averaged toward target)
    """
    target_splits: Dict[Partition, float] = target.to_weighted_splits()
    reference_splits: Dict[Partition, float] = reference.to_weighted_splits()

    intermediate_from_target: Node = calculate_intermediate_tree(
        target, reference_splits
    )
    intermediate_from_reference = calculate_intermediate_tree(reference, target_splits)

    consensus_from_target: Node = calculate_consensus_tree(
        intermediate_from_target, reference_splits
    )
    consensus_from_reference: Node = calculate_consensus_tree(
        intermediate_from_reference, target_splits
    )

    return (
        intermediate_from_target,
        consensus_from_target,
        consensus_from_reference,
        intermediate_from_reference,
    )


def interpolate_adjacent_tree_pairs(tree_list: List[Node]) -> List[Node]:
    """Interpolate between all adjacent pairs in a list of trees."""
    if len(tree_list) < 2:
        raise ValueError("Need at least 2 trees for interpolation")

    logger.info(f"Interpolating {len(tree_list)} trees with adjacent pairs method")
    results: List[Node] = []
    for i in range(len(tree_list) - 1):
        target = tree_list[i]
        reference = tree_list[i + 1]

        trees = interpolate_tree(target, reference)
        results.append(target)
        results.extend(trees)

    results.append(tree_list[-1])
    return results


def build_lattice_interpolation_sequence(
    target: Node, reference: Node, tree_index: int
) -> TreePairInterpolation:
    """
    Build a detailed interpolation sequence between two trees using advanced lattice-based s-edge processing.

    This function implements the core algorithm for morphing one phylogenetic tree topology
    into another through biologically meaningful intermediate states. It uses the jumping
    taxa lattice algorithm to identify structural differences (s-edges) and processes each
    one through a carefully designed 5-step interpolation sequence.

    Algorithm Overview:
    1. **Lattice Edge Discovery**: Uses the jumping taxa algorithm to identify s-edges
       (structural differences) between target and reference trees
    2. **S-edge Ordering**: Sorts s-edges by depth for optimal interpolation progression
    3. **Sequential Processing**: For each s-edge, generates a 5-step morphing sequence:
       - **Down phase**: Apply reference branch weights to target topology
       - **Collapse**: Remove zero-length branches to simplify structure
       - **Reorder**: Match reference tree's node ordering for visual consistency
       - **Pre-snap**: Show reference topology with target weights (contrast)
       - **Snap**: Apply full reference weights to complete the topology change
    4. **Fallback Handling**: When s-edge processing fails, uses classical interpolation
    5. **Data Integration**: Generates synchronized names and tracking for all trees

    The 5-Step Interpolation Process:
    Each s-edge represents a specific structural difference between trees. The 5-step
    process gradually transforms this difference while maintaining tree validity:

    1. **IT{i}_down_{j}** (Down Phase): Applies reference weights within the s-edge
       region while keeping the target topology. This begins the weight transition.

    2. **C{i}_{j}** (Collapse): Removes branches that became zero-length in step 1,
       creating a simplified consensus structure.

    3. **C{i}_{j}_reorder** (Reorder): Rearranges child nodes to match the reference
       tree's ordering, preparing for the topology change.

    4. **IT{i}_up_{j}** (Pre-snap): Shows the reference topology but with target
       weights, creating a visual contrast before the final transition.

    5. **IT{i}_ref_{j}** (Snap): Applies full reference weights to complete the
       topology transformation for this s-edge.

    Naming Convention:
    - IT: Interpolated Tree
    - C: Consensus Tree
    - {i}: Tree index (0-based)
    - {j}: S-edge number within this pair (1-based)
    - Special suffixes: _down, _reorder, _up, _ref, _classical_

    Classical Interpolation Fallback:
    When s-edge processing fails (e.g., s-edge not found in current state), the
    algorithm automatically falls back to classical interpolation, generating
    5 trees named "IT{i}_classical_{j}_1" through "IT{i}_classical_{j}_5".

    Args:
        target: The starting phylogenetic tree for interpolation. Must have
               initialized split indices and consistent taxa with reference.
        reference: The destination tree providing target topology and branch weights.
                  Must have initialized split indices and consistent taxa with target.
        tree_index: Zero-based index used for generating tree names (e.g., 0 for T0→T1 pair).
                   This becomes the {i} component in generated names.

    Returns:
        TreePairInterpolation containing synchronized interpolation data:

        - **trees**: List of interpolated trees (5 trees per s-edge processed)
        - **names**: Descriptive names following the naming convention above
        - **s_edge_tracking**: Partition object for each tree indicating which s-edge
          was processed (None for classical interpolation fallback)
        - **lattice_edge_solutions**: Raw jumping taxa algorithm results mapping
          each s-edge to its solution sets (jumping taxa components)
        - **mapping_one**: Target tree solution-to-atom mappings for analysis
        - **mapping_two**: Reference tree solution-to-atom mappings for analysis
        - **s_edge_distances**: Distance metrics from jumping taxa to s-edges

    Raises:
        Exception: If trees have inconsistent taxa sets or split index initialization fails

    Performance Notes:
        - Time complexity: O(k * m²) where k = number of s-edges, m = number of taxa
        - Generates exactly 5*k interpolation trees where k is the number of s-edges
        - Memory usage proportional to tree size and number of s-edges

    Example:
        # Basic usage
        result = build_lattice_interpolation_sequence(tree1, tree2, 0)
        print(f"Generated {len(result.trees)} interpolation trees")
        print(f"S-edges found: {len(result.lattice_edge_solutions)}")
        print(f"First tree name: {result.names[0]}")  # e.g., "IT0_down_1"

        # Analyze s-edge processing
        for i, (name, s_edge) in enumerate(zip(result.names, result.s_edge_tracking)):
            if s_edge is None:
                print(f"{name}: Classical interpolation")
            else:
                print(f"{name}: Processing s-edge {s_edge.indices}")
    """

    # The lattice algorithm modifies trees during processing, but we need the
    # original unmodified trees for interpolation sequence generation
    target_for_lattice: Node = target.deep_copy()
    reference_for_lattice: Node = reference.deep_copy()

    lattice_edge_solutions: Dict[Partition, List[List[Partition]]] = (
        iterate_lattice_algorithm(target_for_lattice, reference_for_lattice)
    )

    # Process lattice edge data and compute depth-based ordering
    # Depth ordering ensures interpolation progresses from leaves toward root
    lattice_edges: List[Partition] = list(lattice_edge_solutions.keys())
    lattice_edge_data = LatticeEdgeData(lattice_edges, lattice_edge_solutions)
    lattice_edge_data.compute_depths(target, reference)

    logger.info(
        f"Lattice analysis: found {len(lattice_edge_solutions)} s-edges for interpolation"
    )
    if not lattice_edge_solutions:
        logger.warning(
            "No s-edges found - all interpolation will use classical fallback"
        )

    interpolation_trees: List[Node] = []

    # Prepare interpolation parameters
    # Use target tree depths and ascending order for natural progression
    reference_weights = reference.to_weighted_splits()
    target_s_edges: List[Partition] = lattice_edge_data.get_sorted_edges(
        use_reference=False,
        ascending=True,  # Process from leaves toward root
    )

    # Execute the detailed s-edge interpolation sequence generation
    # This creates the 5-step morphing sequence for each s-edge
    sequence_trees, failed_s_edges, tree_names, s_edge_tracking = (
        generate_s_edge_interpolation_sequence(
            target, reference, reference_weights, target_s_edges, tree_index
        )
    )

    interpolation_trees.extend(sequence_trees)

    if failed_s_edges:
        logger.warning(
            f"Classical interpolation used for {len(failed_s_edges)} failed s-edges"
        )

    # Generate solution-to-atom mappings for phylogenetic analysis
    # These mappings connect lattice solutions to specific tree structures
    target_unique_splits: PartitionSet[Partition] = target.to_splits()
    reference_unique_splits: PartitionSet[Partition] = reference.to_splits()

    mapping_one, mapping_two = map_solutions_to_atoms(
        lattice_edge_solutions,
        target_unique_splits,
        reference_unique_splits,
    )

    # Calculate distance metrics from jumping taxa components to their s-edges
    # These metrics help evaluate interpolation quality and complexity
    s_edge_distances: Dict[Partition, Dict[str, float]] = _calculate_s_edge_distances(
        target, reference, lattice_edge_solutions
    )

    return TreePairInterpolation(
        trees=interpolation_trees,
        names=tree_names,
        mapping_one=mapping_one,
        mapping_two=mapping_two,
        s_edge_tracking=s_edge_tracking,
        lattice_edge_solutions=lattice_edge_solutions,
        s_edge_distances=s_edge_distances,
    )


def build_sequential_lattice_interpolations(
    tree_list: List[Node],
) -> TreeInterpolationSequence:
    """
    Build comprehensive lattice-based interpolations between all consecutive tree pairs.

    This is the primary function for creating smooth phylogenetic tree animations using
    advanced lattice-based interpolation. It processes each consecutive pair of trees
    to generate detailed interpolation sequences that morphs from one tree topology
    to another through biologically meaningful intermediate states.

    Algorithm Overview:
    1. **Pair Processing**: For N trees, processes N-1 consecutive pairs (T0→T1, T1→T2, etc.)
    2. **Lattice Analysis**: Uses jumping taxa algorithm to find structural differences (s-edges)
    3. **S-edge Interpolation**: Each s-edge generates a 5-step morphing sequence:
       - Down phase: Apply reference weights to target topology
       - Collapse: Remove zero-length branches for cleaner structure
       - Reorder: Match reference tree's node ordering
       - Pre-snap: Show reference topology with target weights (contrast)
       - Snap: Apply full reference weights (final topology)
    4. **Fallback Handling**: When s-edge processing fails, uses classical interpolation
    5. **Global Assembly**: Combines all pair results into a single coherent sequence

    Tree Sequence Structure:
    - Original trees are inserted at appropriate positions: T0, [interpolated], T1, [interpolated], T2, ...
    - Each tree pair contributes exactly 5*k interpolated trees (k = number of s-edges)
    - Classical fallbacks contribute exactly 5 trees per failed s-edge
    - Global indexing allows direct access to any tree in the complete sequence

    Naming Convention:
    - Original trees: "T0", "T1", "T2", ...
    - Normal interpolation: "IT{i}_down_{j}", "C{i}_{j}", "C{i}_{j}_reorder", "IT{i}_up_{j}", "IT{i}_ref_{j}"
    - Classical fallback: "IT{i}_classical_{j}_1" through "IT{i}_classical_{j}_5"
    - Where i = tree index, j = s-edge number within that pair

    S-edge Tracking:
    - Enables identification of which structural change each interpolated tree represents
    - Original trees: None (no specific s-edge applied)
    - Interpolated trees: Specific Partition object representing the s-edge being processed
    - Classical interpolation: None (doesn't target specific structural changes)

    Args:
        tree_list: List of phylogenetic trees to interpolate between.
                  Must contain at least 2 trees with identical taxa sets.
                  Trees should be rooted and optimized for best results.

    Returns:
        TreeInterpolationSequence containing:
        - **interpolated_trees**: Complete sequence of all trees (originals + interpolated)
        - **interpolation_sequence_labels**: Human-readable names for each tree
        - **mapping_one**: Target tree solution-to-atom mappings for each pair
        - **mapping_two**: Reference tree solution-to-atom mappings for each pair
        - **s_edge_tracking**: S-edge applied for each tree (None for originals/classical)
        - **s_edge_lengths**: Number of interpolation steps generated per pair
        - **lattice_solutions_list**: Raw jumping taxa algorithm results per pair
        - **s_edge_distances_list**: Distance metrics from jumping taxa to s-edges per pair

    Raises:
        ValueError: If tree_list contains fewer than 2 trees

    Performance Notes:
        - Time complexity: O(N * k * m²) where N=pairs, k=avg s-edges per pair, m=taxa count
        - Memory usage: Stores complete interpolation sequence (can be large for many trees)
        - Consider processing in batches for very large tree sets

    Example:
        trees = [tree1, tree2, tree3, tree4]
        result = build_sequential_lattice_interpolations(trees)

        # Access specific trees
        print(f"Total trees generated: {result.total_interpolated_trees}")
        print(f"First interpolated tree: {result.interpolation_sequence_labels[1]}")

        # Analyze pair data
        pair_0_1_data = result.get_pair_data(0)  # T0->T1 interpolation
        print(f"T0->T1 has {pair_0_1_data['s_edge_length']} interpolation steps")

        # Find classical interpolation usage
        classical_indices = result.get_classical_interpolation_indices()
        print(f"Classical interpolation used at indices: {classical_indices}")
    """
    if len(tree_list) < 2:
        raise ValueError("Need at least 2 trees for interpolation")

    logger.info(
        f"Building sequential lattice interpolations for {len(tree_list)} trees ({len(tree_list) - 1} pairs)"
    )
    results: List[Node] = []
    consecutive_tree_names: List[str] = []
    solution_to_atom_mapping_list_one: List[Dict[Partition, Partition]] = []
    solution_to_atom_mapping_list_two: List[Dict[Partition, Partition]] = []
    all_s_edge_tracking: List[Optional[Partition]] = []
    s_edge_lengths: List[int] = []
    lattice_solutions_list: List[Dict[Partition, List[List[Partition]]]] = []
    s_edge_distances_list: List[Dict[Partition, Dict[str, float]]] = []

    for i in range(len(tree_list) - 1):
        target: Node = tree_list[i].deep_copy()
        reference: Node = tree_list[i + 1].deep_copy()

        # Add the current original tree to the sequence
        # Original trees serve as anchor points between interpolation sequences
        results.append(target)
        consecutive_tree_names.append(f"T{i}")
        all_s_edge_tracking.append(None)  # Original tree has no s_edge applied
        logger.debug(
            f"Added original T{i}: sequence_length={len(results)}, names_length={len(consecutive_tree_names)}"
        )

        # Process single tree pair using lattice-based interpolation
        # This generates the detailed 5-step sequence for each s-edge found
        pair_result: TreePairInterpolation = build_lattice_interpolation_sequence(
            target, reference, i
        )
        logger.info(
            f"Processed T{i}→T{i + 1}: generated {len(pair_result.trees)} interpolation trees from {len(pair_result.lattice_edge_solutions)} s-edges"
        )

        # Collect and append all interpolation results from this pair
        # Each pair contributes its interpolated trees to the global sequence
        results.extend(pair_result.trees)
        consecutive_tree_names.extend(pair_result.names)
        solution_to_atom_mapping_list_one.append(pair_result.mapping_one)
        solution_to_atom_mapping_list_two.append(pair_result.mapping_two)
        all_s_edge_tracking.extend(pair_result.s_edge_tracking)
        s_edge_lengths.append(len(pair_result.s_edge_tracking))
        lattice_solutions_list.append(pair_result.lattice_edge_solutions)
        s_edge_distances_list.append(pair_result.s_edge_distances)
        logger.debug(
            f"Extended sequence with T{i}→T{i + 1}: total_trees={len(results)}, total_steps={sum(s_edge_lengths)}"
        )

    # Add the final original tree to complete the sequence
    # This ensures all original trees are represented in the final sequence
    final_tree_index = len(tree_list) - 1
    results.append(tree_list[-1])
    consecutive_tree_names.append(f"T{final_tree_index}")
    all_s_edge_tracking.append(None)  # Final tree has no s_edge applied
    logger.info(
        f"Completed interpolation sequence: {len(results)} total trees from {len(tree_list)} originals + {sum(s_edge_lengths)} interpolated"
    )

    return TreeInterpolationSequence(
        interpolated_trees=results,
        interpolation_sequence_labels=consecutive_tree_names,
        mapping_one=solution_to_atom_mapping_list_one,
        mapping_two=solution_to_atom_mapping_list_two,
        s_edge_tracking=all_s_edge_tracking,
        s_edge_lengths=s_edge_lengths,
        lattice_solutions_list=lattice_solutions_list,
        s_edge_distances_list=s_edge_distances_list,
    )
