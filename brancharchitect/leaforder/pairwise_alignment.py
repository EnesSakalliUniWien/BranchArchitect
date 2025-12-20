from typing import Dict, FrozenSet, List
import logging

from brancharchitect.tree import Node
from brancharchitect.leaforder.tree_order_utils import reorder_tree_if_full_common
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.elements.partition import Partition
from brancharchitect.leaforder.split_analysis import get_common_splits

logger = logging.getLogger(__name__)


def final_pairwise_alignment_pass(trees: List[Node]) -> None:
    """
    Performs a final, sequential alignment pass on a list of trees.

    This function iterates through the trees from left to right. For each pair
    of adjacent trees (T_i, T_{i+1}), it reorders the children of all
    'full-common' nodes in T_{i+1} to match the orientation in T_i.

    This ensures that any shared topology is visually consistent across the
    entire sequence of trees.

    Args:
        trees: The list of trees to be aligned. The trees are modified in-place.
    """
    logger.info("Starting final pairwise alignment pass.")
    if len(trees) < 2:
        return

    for i in range(len(trees) - 1):
        ref_tree = trees[i]
        target_tree = trees[i + 1]

        # Find all splits that are common between the two trees.
        # These are the candidates for reordering.
        common_splits: PartitionSet[Partition] = get_common_splits(
            ref_tree, target_tree
        )

        if not common_splits:
            continue

        # reorder_tree_if_full_common expects a map. We build one for all
        # common splits. The function will internally filter this down to
        # only the 'full-common' ones.
        orientation_map: Dict[Partition, List[FrozenSet[str]]] = {
            split: [] for split in common_splits
        }

        # This function mutates target_tree in place.
        reorder_tree_if_full_common(ref_tree, target_tree, orientation_map)

    logger.info("Final pairwise alignment pass completed.")
