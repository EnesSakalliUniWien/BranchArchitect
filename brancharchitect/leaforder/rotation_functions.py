from typing import Tuple
import logging

# Assuming these imports point to valid modules in your project structure
from brancharchitect.tree import Node
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.elements.partition import Partition
from brancharchitect.leaforder.circular_distances import (
    circular_distance_for_node_subset,
)
from brancharchitect.leaforder.split_analysis import (
    get_unique_splits,
)

logger = logging.getLogger(__name__)


##################################################
#          Local Rotation Tools
##################################################


def try_node_reversal_local(
    node: Node,
    tree: Node,
    initial_dist: float,
    destination_order: Tuple[str, ...],
    rotated_splits: PartitionSet[Partition],
) -> Tuple[bool, float]:
    """
    Swap node's children, measure local (subtree) distance. Revert if no improvement.
    """
    node.swap_children()
    new_dist = circular_distance_for_node_subset(tree, destination_order, node)
    logger.debug(f"  - new_dist: {new_dist}")
    if new_dist < initial_dist:
        rotated_splits.add(node.split_indices)
        return True, new_dist
    else:
        # Revert the swap if there was no improvement.
        node.swap_children()
        return False, initial_dist


def optimize_splits(
    tree: Node,
    splits_to_optimize: PartitionSet[Partition],
    destination_order: Tuple[str, ...],
    rotated_splits: PartitionSet[Partition],
) -> bool:
    """
    For each split in splits_to_optimize, attempt a local reversal.
    Returns True if any improvement was made.
    """
    any_improvement = False
    logger.debug(f"Optimizing {len(splits_to_optimize)} splits")
    for sp in splits_to_optimize:
        node = tree.find_node_by_split(sp)
        if node and node.children:
            init_dist = circular_distance_for_node_subset(tree, destination_order, node)
            logger.debug(f"- Optimizing split {sp} with initial_dist: {init_dist}")
            improved, _ = try_node_reversal_local(
                node, tree, init_dist, destination_order, rotated_splits
            )
            if improved:
                any_improvement = True
    return any_improvement


def optimize_unique_splits(
    tree1: Node,
    tree2: Node,
    destination_order: tuple[str, ...],
    rotated_splits: PartitionSet[Partition] = PartitionSet(),
) -> bool:
    """
    For each 'unique' split in tree2, attempt a local reversal.
    """
    # Step 1: Find all splits that are in tree2 but not tree1.
    unique2: PartitionSet[Partition] = get_unique_splits(tree1, tree2)

    # Step 2: Try to improve tree2 by reversing the children of each of those unique splits.
    return optimize_splits(tree2, unique2, destination_order, rotated_splits)
