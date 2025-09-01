from typing import Literal, Tuple, Dict

# Assuming these imports point to valid modules in your project structure
from brancharchitect.tree import Node
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.elements.partition import Partition
from brancharchitect.leaforder.circular_distances import (
    circular_distance_for_node_subset,
    circular_distance_based_on_reference,
)
from brancharchitect.jumping_taxa.lattice.iterate_lattice_algorithm import (
    iterate_lattice_algorithm,
)

##################################################
#          Local Rotation Tools
##################################################

# Global cache for split set operations, with a precise type hint for the key.
_split_pair_cache: Dict[Tuple[int, int, str], PartitionSet[Partition]] = {}


def get_unique_splits(tree1: Node, tree2: Node) -> PartitionSet[Partition]:
    """
    Returns the set of splits that are in tree2 but not in tree1.
    Uses a cache for repeated calls on the same tree pair.
    """
    key: Tuple[int, int, Literal["unique"]] = (id(tree1), id(tree2), "unique")
    if key in _split_pair_cache:
        return _split_pair_cache[key]

    s1: PartitionSet[Partition] = tree1.to_splits()
    s2: PartitionSet[Partition] = tree2.to_splits()
    result: PartitionSet[Partition] = s2 - s1
    _split_pair_cache[key] = result
    return result


def get_common_splits(tree1: Node, tree2: Node) -> PartitionSet[Partition]:
    """
    Returns the set of splits that are common to both tree1 and tree2.
    Uses a cache for repeated calls on the same tree pair.
    """
    # Using Literal in the type hint improves static analysis.
    key: Tuple[int, int, Literal["common"]] = (id(tree1), id(tree2), "common")
    if key in _split_pair_cache:
        return _split_pair_cache[key]

    s1: PartitionSet[Partition] = tree1.to_splits()
    s2: PartitionSet[Partition] = tree2.to_splits()
    result: PartitionSet[Partition] = s1 & s2
    _split_pair_cache[key] = result
    return result


def get_s_edge_splits(tree1: Node, tree2: Node) -> PartitionSet[Partition]:
    """
    Returns 's-edge' splits: common splits in tree2 where children differ.
    Uses a cache for repeated calls on the same tree pair.
    """
    key: Tuple[int, int, Literal["sedge"]] = (id(tree1), id(tree2), "sedge")
    # CORRECTED: Added cache check to prevent redundant calculations.
    if key in _split_pair_cache:
        return _split_pair_cache[key]
    tree1_copy = tree1.deep_copy()
    tree2_copy = tree2.deep_copy()
    # This calculation is now only performed if the result is not in the cache.
    # Use deep copies to prevent the lattice algorithm from modifying the original trees
    s_edge_solutions = iterate_lattice_algorithm(tree1_copy, tree2_copy)
    s_edges_list = list(s_edge_solutions.keys())
    # Convert to PartitionSet for consistency with function signature
    s_edges_set: PartitionSet[Partition] = PartitionSet(
        set(s_edges_list), encoding=tree1.taxa_encoding
    )
    _split_pair_cache[key] = s_edges_set
    return s_edges_set


def clear_split_pair_cache() -> None:
    """
    Clear the global split pair cache. Call this after any tree mutation.
    """
    _split_pair_cache.clear()


def try_node_reversal_local(
    node: Node,
    tree: Node,
    initial_dist: float,
    reference_order: Tuple[str, ...],
    rotated_splits: PartitionSet[Partition],
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
        # Revert the swap if there was no improvement.
        node.swap_children()
        return False, initial_dist


def optimize_splits(
    tree: Node,
    splits_to_optimize: PartitionSet[Partition],
    reference_order: Tuple[str, ...],
    rotated_splits: PartitionSet[Partition],
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


def _test_and_revert_flip(
    node: Node, tree: Node, reference_order: Tuple[str, ...], _: str
) -> float:
    """
    Helper to flip node's children, evaluate distance, and revert.
    The final parameter is ignored, kept for potential future use or API consistency.
    """
    node.swap_children()
    dist = circular_distance_based_on_reference(tree, reference_order)
    # Revert swap. Using swap_children() again is safer than direct assignment.
    node.swap_children()
    return dist


def _apply_best_flip(node: Node, best_type: str) -> None:
    """Helper to apply the best flip for real."""
    # Currently, both flip types result in the same action.
    if best_type in ("global", "local"):
        node.swap_children()


def optimize_unique_splits(
    tree1: Node,
    tree2: Node,
    reference_order: tuple[str, ...],
    rotated_splits: PartitionSet[Partition] = PartitionSet(),
) -> bool:
    """
    For each 'unique' split in tree2, attempt a local reversal.
    """
    # Step 1: Find all splits that are in tree2 but not tree1.
    unique2: PartitionSet[Partition] = get_unique_splits(tree1, tree2)

    # Step 2: Try to improve tree2 by reversing the children of each of those unique splits.
    return optimize_splits(tree2, unique2, reference_order, rotated_splits)


def optimize_s_edge_splits(
    tree1: Node,
    tree2: Node,
    reference_order: Tuple[str, ...],
    rotated_splits: PartitionSet[Partition],
) -> bool:
    """ """
    s_edges_sorted = get_s_edge_splits(tree1, tree2)
    if not s_edges_sorted:
        return False

    current_dist = circular_distance_based_on_reference(tree2, reference_order)
    any_improvement = False

    for sp in s_edges_sorted:
        node = tree2.find_node_by_split(sp)
        if node and node.children:
            # Test both 'global' and 'local' flips to see which is better.
            # The flip_type parameter is unused in _test_and_revert_flip but passed for clarity.
            dist_g = _test_and_revert_flip(node, tree2, reference_order, "global")
            dist_l = _test_and_revert_flip(node, tree2, reference_order, "local")

            best_flip_dist = min(dist_g, dist_l)

            if best_flip_dist < current_dist:
                best_type = "global" if dist_g < dist_l else "local"
                _apply_best_flip(node, best_type)
                current_dist = best_flip_dist
                any_improvement = True
                rotated_splits.add(sp)

    return any_improvement
