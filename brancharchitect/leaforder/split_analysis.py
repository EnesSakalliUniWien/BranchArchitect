import logging
from functools import lru_cache

# Assuming these imports point to valid modules in your project structure
from brancharchitect.tree import Node
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.elements.partition import Partition
from brancharchitect.jumping_taxa.lattice.iterate_lattice_algorithm import (
    iterate_lattice_algorithm,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=128)
def get_unique_splits(tree1: Node, tree2: Node) -> PartitionSet[Partition]:
    """
    Returns the set of splits that are in tree2 but not in tree1.
    Uses LRU cache for repeated calls on the same tree pair.
    """
    s1: PartitionSet[Partition] = tree1.to_splits()
    s2: PartitionSet[Partition] = tree2.to_splits()
    return s2 - s1


@lru_cache(maxsize=128)
def get_common_splits(tree1: Node, tree2: Node) -> PartitionSet[Partition]:
    """
    Returns the set of splits that are common to both tree1 and tree2.
    Uses LRU cache for repeated calls on the same tree pair.
    """
    s1: PartitionSet[Partition] = tree1.to_splits()
    s2: PartitionSet[Partition] = tree2.to_splits()
    return s1 & s2


@lru_cache(maxsize=64)
def get_active_changing_splits(tree1: Node, tree2: Node) -> PartitionSet[Partition]:
    """
    Returns active changing splits: common splits in tree2 where children differ.
    Uses LRU cache for repeated calls on the same tree pair.
    """
    tree1_copy = tree1.deep_copy()
    tree2_copy = tree2.deep_copy()
    # Use deep copies to prevent the lattice algorithm from modifying the original trees
    # iterate_lattice_algorithm returns a tuple: (dict, list)
    active_changing_split_solutions, _ = iterate_lattice_algorithm(
        tree1_copy, tree2_copy
    )
    active_changing_splits_list = list(active_changing_split_solutions.keys())
    # Convert to PartitionSet for consistency with function signature
    active_changing_splits_set: PartitionSet[Partition] = PartitionSet(
        set(active_changing_splits_list), encoding=tree1.taxa_encoding
    )
    return active_changing_splits_set


def get_common_splits_without_descendant_changes(
    source_tree: Node, destination_tree: Node
) -> PartitionSet[Partition]:
    """
    Returns the set of splits that are common to both tree1 and tree2,
    excluding those where the child order differs due to descendant changes.
    """
    common_splits: PartitionSet[Partition] = get_common_splits(
        source_tree, destination_tree
    )
    filtered_splits: PartitionSet[Partition] = PartitionSet()

    for sp in common_splits:
        source_node = source_tree.find_node_by_split(sp)

        destinsation_node = destination_tree.find_node_by_split(sp)

        # Skip if either node is not found
        if source_node is None or destinsation_node is None:
            continue

        source_node_splits = source_node.to_splits()

        destinsation_node_splits = destinsation_node.to_splits()

        source_node_splits_unique = source_node_splits - destinsation_node_splits

        destinsation_node_splits_unique = destinsation_node_splits - source_node_splits

        if not source_node_splits_unique and not destinsation_node_splits_unique:
            filtered_splits.add(sp)

    return filtered_splits


def clear_split_pair_cache() -> None:
    """
    Clear the LRU caches. Call this after any tree mutation.
    """
    get_unique_splits.cache_clear()
    get_common_splits.cache_clear()
    get_active_changing_splits.cache_clear()
