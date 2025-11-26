"""Core type definitions for phylogenetic analysis."""

from typing import TypedDict, Optional


class TreeMetadata(TypedDict):
    """Global metadata for each tree in the complete interpolation sequence."""

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

    source_tree_global_index: Optional[int]
    """Global index of the source tree this interpolated tree comes FROM (None for original trees)."""
