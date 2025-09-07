from typing import List

from brancharchitect.tree import Node
from brancharchitect.leaforder.rotation_functions import (
    get_common_splits,
    clear_split_pair_cache,
)
from brancharchitect.leaforder.tree_order_utils import (
    build_orientation_map,
    reorder_tree_if_full_common,
)


def final_pairwise_alignment_pass(trees: List[Node]) -> None:
    """
    Perform a final alignment pass to ensure consistent orientation
    of common subtrees across adjacent pairs of trees.

    Modifies the list of trees in-place.
    """
    if len(trees) < 2:
        return

    for i in range(len(trees) - 1):
        tree1 = trees[i]
        tree2 = trees[i + 1]

        common_splits_pair = get_common_splits(tree1, tree2)
        if not common_splits_pair:
            continue

        orientation_map = build_orientation_map(tree1, common_splits_pair)
        reorder_tree_if_full_common(tree1, tree2, orientation_map)

    clear_split_pair_cache()

