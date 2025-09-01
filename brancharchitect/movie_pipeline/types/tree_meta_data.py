"""Core type definitions for phylogenetic analysis."""

from typing import List, TypedDict, Optional


class TreeMetadata(TypedDict):
    """
    Global metadata for each tree in the complete interpolation sequence.

    Provides JSON-serializable lookup information for every tree, enabling
    easy navigation between interpolated trees and their source data.
    All indices are global across the entire interpolation sequence.

    Example usage:
        # Find which tree pair generated tree at index 15
        metadata = result.tree_metadata[15]
        if metadata.tree_pair_key:
            pair_solution = result.tree_pair_solutions[metadata.tree_pair_key]

        # Check if this is step 3 of interpolation pair 2_3
        if metadata.step_in_pair == 3 and metadata.tree_pair_key == "pair_2_3":
            # This is the reordering step for trees 2->3
    """

    # Global identification across entire sequence
    global_tree_index: int
    """Index of this tree in the flattened interpolated_trees list (0-based).

    This provides direct access: interpolated_trees[global_tree_index] gets this tree.
    Increments continuously across all tree pairs: 0, 1, 2, ..., N-1
    """

    tree_name: str
    """Human-readable name for this tree.

    Format examples:
    - "T0", "T1", "T2" for original trees
    - "IT0_down_1" for down phase of s-edge 1 in pair 0->1
    - "C0_1" for collapse phase of s-edge 1 in pair 0->1
    - "C0_1_reorder" for reorder phase of s-edge 1 in pair 0->1
    - "IT0_up_1" for up phase of s-edge 1 in pair 0->1
    - "IT0_ref_1" for reference snap of s-edge 1 in pair 0->1
    - "IT0_classical_1_3" for classical fallback step 3 of s-edge 1 in pair 0->1
    """

    # Source and relationship tracking
    source_tree_index: Optional[int]
    """Index of the original source tree (0-based), or None for interpolated trees.

    - For original trees: matches their position in input list (0, 1, 2, ...)
    - For interpolated trees: None (they don't correspond to a single source)

    Example: All interpolation steps between T1 and T2 have source_tree_index=None
    """

    tree_pair_key: Optional[str]
    """Key to lookup the TreePairSolution that generated this tree, or None for original trees.

    Format: "pair_{source_idx}_{target_idx}"
    - "pair_0_1": Interpolation from tree 0 to tree 1
    - "pair_1_2": Interpolation from tree 1 to tree 2
    - None: This is an original tree, not interpolated

    Use: tree_pair_solutions[tree_pair_key] gets the full solution data
    """

    # Interpolation step context
    s_edge_tracker: Optional[List[int]]
    """Indices of the s-edge being processed, or None for original trees.

    Contains the list of taxa indices that define the s-edge partition being
    processed during this interpolation step. None for original trees that
    are not part of the interpolation.

    Format: List of integer indices (e.g., [1, 3, 5]) or None for original trees
    Use: Track which phylogenetic split is being modified
    """

    step_in_pair: Optional[int]
    """Step number within the s-edge interpolation (1-5), or None for original trees.

    Each s-edge generates exactly 5 interpolation steps:
    - 1: Down phase (apply reference weights to s-edge subset)
    - 2: Collapse (remove zero-length branches from consensus)
    - 3: Reorder (match reference tree's node ordering)
    - 4: Up phase (pre-snap with reference topology but target weights)
    - 5: Snap (final reference state with graft operation)

    None: This is an original tree, not an interpolation step

    Note: step_in_pair refers to position within ONE s-edge, not the entire pair.
    """

    subtree_tracker: Optional[List[int]]
    """List of indices for the subtree being processed, or None.

    Direct representation of the Partition indices that define the subtree region
    being modified to generate this tree. None for original trees or when subtree
    tracking is not available.

    Format: List of integer indices, e.g., [2,4,6]
    Use: Track which subtree region is being modified at each step
    """
