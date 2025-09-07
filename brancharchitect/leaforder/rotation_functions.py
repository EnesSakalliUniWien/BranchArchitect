from typing import Literal, Tuple, Dict
import logging

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
from brancharchitect.jumping_taxa.lattice.depth_computation import (
    compute_lattice_edge_depths,
)

logger = logging.getLogger(__name__)

# Private feature flag: enable impact-aware tie-breaker in s-edge ordering (experiments only)
_USE_IMPACT_TIEBREAKER: bool = True

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
    """Try local flips at s-edges in a deterministic, subset-aware order.

    Order aligns with interpolation: subsets first, then smaller partitions,
    then tree depth as a fine-grained tiebreaker.
    """
    s_edges_set = get_s_edge_splits(tree1, tree2)
    if not s_edges_set:
        return False

    # Build deterministic order consistent with interpolation, with optional impact tie-breaker
    s_edges: list[Partition] = list(s_edges_set)
    depth_map = compute_lattice_edge_depths(s_edges, tree2)

    impact_map: Dict[Partition, float] = {}
    if _USE_IMPACT_TIEBREAKER:
        for sp in s_edges:
            node = tree2.find_node_by_split(sp)
            if node is not None:
                impact_map[sp] = circular_distance_for_node_subset(tree2, reference_order, node)
            else:
                impact_map[sp] = 0.0

    def _sort_key(sp: Partition):
        depth = depth_map.get(sp, 0)
        size = len(tuple(sp))
        idxs = tuple(int(i) for i in sp)
        if _USE_IMPACT_TIEBREAKER:
            return (depth, -impact_map.get(sp, 0.0), size, idxs)
        return (depth, size, idxs)

    s_edges.sort(key=_sort_key)

    order_indices = [tuple(int(i) for i in sp) for sp in s_edges]
    if _USE_IMPACT_TIEBREAKER:
        impacts_dbg = [impact_map.get(sp, 0.0) for sp in s_edges]
        logger.info(
            f"Optimizer s-edge order (impact-aware): {order_indices} with impacts {impacts_dbg}"
        )
    else:
        logger.info(f"Optimizer s-edge order: {order_indices}")
    logger.info(f"Processing {len(s_edges)} s-edges for optimization")

    current_dist = circular_distance_based_on_reference(tree2, reference_order)
    any_improvement = False

    for sp in s_edges:
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
                improvement = current_dist - best_flip_dist
                logger.debug(f"Applied {best_type} flip for s-edge {sp}, improvement: {improvement:.4f}")
                current_dist = best_flip_dist
                any_improvement = True
                rotated_splits.add(sp)

    return any_improvement
