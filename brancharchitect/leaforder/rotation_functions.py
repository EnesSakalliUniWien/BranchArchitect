from typing import Tuple
from brancharchitect.tree import Node
from brancharchitect.partition_set import PartitionSet
from brancharchitect.leaforder.circular_distances import (
    circular_distance_for_node_subset,
    circular_distance_based_on_reference,
)
##################################################
#          Local Rotation Tools
##################################################

# Global cache for split set operations
_split_pair_cache = {}

def get_unique_splits(tree1: Node, tree2: Node) -> PartitionSet:
    """
    Returns the set of splits that are in tree2 but not in tree1.
    Uses a cache for repeated calls on the same tree pair.
    """
    key = (id(tree1), id(tree2), 'unique')
    if key in _split_pair_cache:
        return _split_pair_cache[key]
    s1 = tree1.to_splits()
    s2 = tree2.to_splits()
    result = s2 - s1
    _split_pair_cache[key] = result.atom()
    return result

def get_common_splits(tree1: Node, tree2: Node) -> PartitionSet:
    """
    Returns the set of splits that are common to both tree1 and tree2.
    Uses a cache for repeated calls on the same tree pair.
    """
    key = (id(tree1), id(tree2), 'common')
    if key in _split_pair_cache:
        return _split_pair_cache[key]
    s1 = tree1.to_splits()
    s2 = tree2.to_splits()
    result = s1 & s2
    _split_pair_cache[key] = result
    return result

def get_s_edge_splits(tree1: Node, tree2: Node) -> PartitionSet:
    """
    Returns the set of 's-edge' splits: common splits in tree2 where at least one child is unique to tree2.
    Uses a cache for repeated calls on the same tree pair.
    """
    key = (id(tree1), id(tree2), 'sedge')
    if key in _split_pair_cache:
        return _split_pair_cache[key]
    s1 = tree1.to_splits()
    s2 = tree2.to_splits()
    common = s1 & s2
    unique2 = s2 - s1
    s_edges = PartitionSet()
    for sp in common:
        node = tree2.find_node_by_split(sp)
        if node and node.children:
            if any(ch.split_indices in unique2 for ch in node.children if ch.split_indices):
                s_edges.add(sp)
    _split_pair_cache[key] = s_edges
    return s_edges

def clear_split_pair_cache():
    """
    Clear the global split pair cache. Call this after any tree mutation.
    """
    _split_pair_cache.clear()

def try_node_reversal_global(
    node: Node,
    tree: Node,
    initial_dist: float,
    reference_order: Tuple[str, ...],
    rotated_splits: PartitionSet,
):
    """
    Swap node's children, measure full-tree distance vs. reference_order.
    Revert if no improvement; otherwise record node.split in rotated_splits.
    """
    node.swap_children()
    new_dist = circular_distance_based_on_reference(tree, reference_order)
    if new_dist < initial_dist:
        rotated_splits.add(node.split_indices)
        return True, new_dist
    else:
        node.swap_children()
        return False, initial_dist


def try_node_reversal_local(
    node: Node,
    tree: Node,
    initial_dist: float,
    reference_order: Tuple[str, ...],
    rotated_splits: PartitionSet,
) -> Tuple[bool, float]:
    """
    Swap node's children, measure local (subtree) distance. Revert if no improvement.
    """
    node.swap_children()
    new_dist = circular_distance_for_node_subset(tree, reference_order, node)
    if new_dist < initial_dist:
        rotated_splits.add(node.split_indices)
        return True, new_dist
    else:
        node.swap_children()
        return False, new_dist


def optimize_splits(
    tree: Node,
    splits_to_optimize: PartitionSet,
    reference_order: tuple,
    rotated_splits: PartitionSet,
) -> bool:
    """
    For each split in splits_to_optimize, attempt a local reversal.
    Returns True if any improvement was made.
    """
    any_improvement = False
    for sp in splits_to_optimize:
        node = tree.find_node_by_split(sp)
        if node and node.children:
            init_dist = circular_distance_for_node_subset(tree, reference_order, node)
            improved, _ = try_node_reversal_local(
                node, tree, init_dist, reference_order, rotated_splits
            )
            if improved:
                any_improvement = True
    return any_improvement


def optimize_unique_splits(
    tree1: Node,
    tree2: Node,
    reference_order: Tuple[str, ...],
    rotated_splits: PartitionSet,
) -> bool:
    """
    For each 'unique' split in tree2, attempt a local reversal.
    """
    unique2 = get_unique_splits(tree1, tree2)
    return optimize_splits(tree2, unique2, reference_order, rotated_splits)


def _test_and_revert_flip(node: Node, tree: Node, reference_order, flip_type: str) -> float:
    """
    Helper to flip node's children, evaluate distance, revert, and invalidate cache.
    flip_type: 'global' or 'local' (for clarity, but both do the same here)
    Returns the computed distance after flip.
    """
    node.swap_children()
    dist = circular_distance_based_on_reference(tree, reference_order)
    # revert
    node.children = node.children[::-1]  # swap back
    node.invalidate_caches()  # Invalidate cache after direct assignment
    return dist

def _apply_best_flip(node: Node, best_type: str):
    """
    Helper to apply the best flip (global or local) for real.
    """
    if best_type in ('global', 'local'):
        node.swap_children()


def optimize_s_edge_splits(tree1: Node, tree2: Node, reference_order, rotated_splits):
    s_edges_sorted = get_s_edge_splits(tree1, tree2)
    current_dist = circular_distance_based_on_reference(tree2, reference_order)
    any_improvement = False

    for sp in s_edges_sorted:
        node = tree2.find_node_by_split(sp)
        if node and node.children:
            original_children = node.children[:]

            # Test both flips
            dist_g = _test_and_revert_flip(node, tree2, reference_order, 'global')
            dist_l = _test_and_revert_flip(node, tree2, reference_order, 'local')

            # Compare
            best_flip_dist = min(dist_g, dist_l, current_dist)
            if best_flip_dist < current_dist:
                if dist_g < dist_l:
                    best_type = 'global'
                    new_dist = dist_g
                else:
                    best_type = 'local'
                    new_dist = dist_l
                _apply_best_flip(node, best_type)
                current_dist = new_dist
                any_improvement = True
                rotated_splits.add(sp)
            else:
                node.children = original_children[:]
                node.invalidate_caches()  # Invalidate cache after direct assignment
    return any_improvement