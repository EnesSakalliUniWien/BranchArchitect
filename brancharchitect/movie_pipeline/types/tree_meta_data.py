"""Core type definitions for phylogenetic analysis."""

from typing import TypedDict, Optional


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

    tree_pair_key: Optional[str]
    """Key to lookup the TreePairSolution that generated this tree, or None for original trees.

    Format: "pair_{source_idx}_{target_idx}"
    - "pair_0_1": Interpolation from tree 0 to tree 1
    - "pair_1_2": Interpolation from tree 1 to tree 2
    - None: This is an original tree, not interpolated

    Use: tree_pair_solutions[tree_pair_key] gets the full solution data
    """

    # Interpolation step context
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

    # Reduced metadata for lean payloads: other trackers removed from TreeMetadata
