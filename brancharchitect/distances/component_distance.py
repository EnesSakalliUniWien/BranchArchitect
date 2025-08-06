# =============================================================================
"""
Component Distance Module

This module provides functionality for computing component-based distances between phylogenetic trees.
It includes optimized core functions that work directly with Partition objects, as well as adapter
functions for backward compatibility with tuple-based interfaces.

Main functions:
- component_distance: Compute component distances between two trees
- jump_distance: Compute jump distance for a single component
- jump_path_distance: Compute weighted jump path distances
- calculate_component_distances: Analyze distances across multiple trees
"""

# Standard library imports
import numpy as np
from collections import Counter
from typing import List, Tuple, Optional

# Local imports
from brancharchitect.tree import Node
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.elements.partition import Partition



# =============================================================================
# JUMP PATH FROM COMPONENT TO S_EDGE NODE
# =============================================================================
def jump_path_component_to_s_edge(
    tree: Node, component: Partition, s_edge_split: Partition
) -> List[Node]:
    """
    Compute the jump path from the node corresponding to the component (solution taxa)
    up to the node corresponding to the s_edge split in the tree.

    Args:
        tree: The tree (Node) to search in.
        component: The Partition representing the solution taxa (component).
        s_edge_split: The Partition representing the s_edge split (target split).

    Returns:
        List[Node]: The path from the component node up to (and including) the s_edge node.
        If the component node is not a descendant of the s_edge node, returns an empty list.
    """
    start_node: Node | None = tree.find_node_by_split(component)
    target_node: Node | None = tree.find_node_by_split(s_edge_split)
    if start_node is None or target_node is None:
        return []
    # Traverse up from start_node to target_node
    path: list[Node] = []
    current: Node = start_node
    while current is not None:
        path.append(current)
        if current is target_node:
            return path  # Path from component up to s_edge (inclusive)
        current = getattr(current, "parent", None)
    # If we reach here, target_node was not found in the ancestry
    return []


# =============================================================================
# CORE PERFORMANCE-OPTIMIZED FUNCTIONS
# =============================================================================


def _component_distance_core(
    tree1: Node, tree2: Node, components: List[Partition], weighted: bool = False
) -> List[float]:
    """
    Core function that computes component distances using Partition objects directly.
    This is the performance-optimized version that avoids conversions.
    """
    distances: List[float] = []
    tree1_splits: PartitionSet[Partition] = tree1.to_splits()
    tree2_splits: PartitionSet[Partition] = tree2.to_splits()

    for component in components:
        d1: float = _jump_distance_core(
            tree1, tree2_splits, component, weighted=weighted
        )
        d2: float = _jump_distance_core(
            tree2, tree1_splits, component, weighted=weighted
        )
        distances.append(d1 + d2)
    return distances


def _jump_distance_core(
    node: Node,
    reference: PartitionSet[Partition],
    component: Partition,
    weighted: bool = False,
) -> float:
    """
    Core function that computes jump distance using Partition objects directly.
    This is the performance-optimized version that avoids conversions.
    """
    path: List[Node] = jump_path(node, reference, component)
    if weighted:
        return float(sum(n.length for n in path if n.length is not None))
    else:
        return float(len(path))


def _jump_path_distance_core(
    tree1: Node, tree2: Node, components: List[Partition], weighted: bool = False
) -> List[float]:
    """
    Core function that computes jump path distances using Partition objects directly.
    This is the performance-optimized version that avoids conversions.
    """
    paths1 = [
        jump_path(tree1, tree2.to_splits(), component) for component in components
    ]
    paths2 = [
        jump_path(tree2, tree1.to_splits(), component) for component in components
    ]

    counter = Counter(node.split_indices for path in paths1 + paths2 for node in path)

    distances: List[float] = []
    for p1, p2 in zip(paths1, paths2):
        if weighted:
            d = sum(
                (n.length if n.length is not None else 0.0) / counter[n.split_indices]
                for n in p1 + p2
            )
        else:
            d = sum(1.0 / counter[n.split_indices] for n in p1 + p2)
        distances.append(d)
    return distances


# =============================================================================
# PATH COMPUTATION FUNCTIONS
# =============================================================================


# --- Helper: Memoization cache for jump_path ---
def _get_jump_path_cache() -> dict[tuple[int, int, int], list["Node"]]:
    """Get or initialize the memoization cache for jump_path."""
    if not hasattr(jump_path, "__cache"):
        jump_path.__cache = {}  # type: ignore[attr-defined]
    return jump_path.__cache  # type: ignore[attr-defined]


# --- Helper: Get bitmask for a node's split_indices ---
def _get_node_bitmask(node: Node) -> int:
    """Get the bitmask for a node's split_indices, with fallback."""
    try:
        return node.split_indices.bitmask
    except AttributeError:
        return hash(node.split_indices)


# --- Helper: Find the child node whose split contains the target component ---
def _find_child_with_component(
    children: List[Node], target_bitmask: int
) -> Optional[Node]:
    """Find the child node whose split contains the target component."""
    for child in children:
        child_bitmask = _get_node_bitmask(child)
        # If all bits in component are in child.split_indices (component is subset of child)
        if (target_bitmask & child_bitmask) == target_bitmask:
            return child
    return None


# --- Helper: Main loop to build the jump path ---
def _build_jump_path_main(
    node: Node, reference: PartitionSet[Partition], target_bitmask: int
) -> List[Node]:
    """Build the jump path from node to target component."""
    path: List[Node] = []
    current_node = node
    while True:
        current_bitmask = _get_node_bitmask(current_node)
        # Stop if we've reached the target
        if current_bitmask == target_bitmask:
            break
        # If current split is in reference, clear the path (reset)
        if current_node.split_indices in reference:
            path.clear()
        else:
            path.append(current_node)
        # Find the next child containing the component
        next_node = _find_child_with_component(current_node.children, target_bitmask)
        if next_node is None:
            break  # Component is not actually a component
        current_node: Node = next_node
    return path


# --- Main API: jump_path ---
def jump_path(
    node: Node, reference: PartitionSet[Partition], component: Partition
) -> List[Node]:
    """
    Compute the jump path for a component with memoization for performance.

    Note:
        This function computes the path from the root node down to the node whose split matches the component.
        If you want the path from a component node up to a specific s_edge node, use `jump_path_component_to_s_edge`.
    """
    cache: dict[tuple[int, int, int], list[Node]] = _get_jump_path_cache()
    component_bitmask: int = component.bitmask
    key: Tuple[int] = (id(node), id(reference), component_bitmask)
    # Check cache first
    cached_result = cache.get(key)
    if cached_result is not None:
        return list(cached_result)
    # Build the path using helpers
    path: List[Node] = _build_jump_path_main(node, reference, component_bitmask)
    # Cache as tuple to avoid mutation issues
    cache[key] = tuple(path)
    return list(path)


# =============================================================================
# PUBLIC API FUNCTIONS (Adapters for backward compatibility)
# =============================================================================


def component_distance(
    tree1: Node, tree2: Node, components: List[Tuple[str, ...]], weighted: bool = False
) -> List[float]:
    """
    Adapter function that converts tuple-of-strings to Partition objects
    and delegates to the core performance function.
    """
    # Convert components once at the beginning
    component_partitions: List[Partition] = [
        tree1._index(component) for component in components
    ]
    return _component_distance_core(
        tree1, tree2, component_partitions, weighted=weighted
    )


def jump_distance(
    node: Node,
    reference: PartitionSet[Partition],
    component: Tuple[str, ...],
    weighted: bool = False,
) -> float:
    """
    Adapter function that converts tuple-of-strings to Partition objects
    and delegates to the core performance function.
    """
    component_partition: Partition = node._index(component)
    return _jump_distance_core(node, reference, component_partition, weighted=weighted)


def jump_path_distance(
    tree1: Node, tree2: Node, components: List[Tuple[str, ...]], weighted: bool = False
) -> List[float]:
    """
    Adapter function that converts tuple-of-strings to Partition objects
    and delegates to the core performance function.
    """
    # Convert component tuples to Partition objects once at the beginning
    components_partitions: List[Partition] = [tree1._index(c) for c in components]
    return _jump_path_distance_core(
        tree1, tree2, components_partitions, weighted=weighted
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def clear_jump_path_cache() -> None:
    """Clear the memoization cache for jump_path."""
    if hasattr(jump_path, "__cache"):
        jump_path.__cache.clear()  # type: ignore[attr-defined]


def extract_leaf_order(trees: List[Node]) -> List[str]:
    """
    Extract a canonical leaf order from the first tree (or consensus of all trees).
    Assumes all trees have the same set of leaves.
    """
    if not trees:
        raise ValueError("No trees provided.")
    return [leaf.name for leaf in trees[0].get_leaves()]


def get_lattice_solution_sizes(
    tree1: Node, tree2: Node, leaf_order: List[str]
) -> List[int]:
    """
    Run the iterative lattice algorithm and return the sizes of all minimal reconciliation solutions.
    Returns an empty list if no solutions are found.
    """
    from brancharchitect.jumping_taxa.lattice.lattice_solver import iterate_lattice_algorithm

    s_edge_solutions = iterate_lattice_algorithm(tree1, tree2, leaf_order)
    if not s_edge_solutions:
        return []
    # Flatten all solution sets to get all solutions
    all_solutions = [
        sol for solution_sets in s_edge_solutions.values() for sol in solution_sets
    ]
    return [len(sol) for sol in all_solutions]


# =============================================================================
# HIGH-LEVEL ANALYSIS FUNCTIONS
# =============================================================================


def calculate_component_distance_matrix(
    trees: List[Node],
    list_of_components: List[List[Partition]],
    weighted: bool = False,
):
    """
    Analyze component-based distances between trees using component_distance.py.

    Args:
        trees: List of phylogenetic trees as Node objects.
        list_of_components: List of component lists (each as Partition objects).
        weighted: Use weighted component distances.

    Returns:
        numpy.ndarray: Distance matrix between trees.
    """
    num_trees: int = len(trees)
    distance_matrix = np.zeros((num_trees, num_trees), dtype=float)

    for i in range(num_trees):
        for j in range(num_trees):
            for components in list_of_components:
                if i == j:
                    distance_matrix[i, j] = 0.0
                else:
                    dists = _component_distance_core(
                        trees[i], trees[j], components=components, weighted=weighted
                    )
                    distance_matrix[i, j] = np.mean(dists) if dists else 0.0
    return distance_matrix


def calculate_normalised_matrix(results, max_trees):
    max_component_sum = max(
        (sum(len(c) for c in comp) for _, _, comp, _, _, _ in results), default=1
    )
    max_num_solutions = max((len(comp) for _, _, comp, _, _, _ in results), default=1)
    max_path_length = max(
        ((len(pi) + len(pj)) for _, _, _, _, pi, pj in results), default=1
    )
    # 2. Optionally set weights for each component (can be tuned)
    w1, w2, w3 = 1.0, 1.0, 1.0
    # 3. Build normalized distance matrix
    normalized_matrix = np.zeros((max_trees, max_trees), dtype=float)
    for i, j, components, s_edges, path_i, path_j in results:
        component_sum: int = sum(len(c) for c in components)
        num_solutions: int = len(components)
        path_lengths: int = sum([len(pi) + len(pj) for pi, pj in zip(path_i, path_j)])
        norm_component_sum: float | Literal[0] = (
            component_sum / max_component_sum if max_component_sum else 0
        )
        norm_num_solutions: float | Literal[0] = (
            num_solutions / max_num_solutions if max_num_solutions else 0
        )
        norm_path_length: float | Literal[0] = (
            path_lengths / max_path_length if max_path_length else 0
        )
        dist = w1 * norm_component_sum + w2 * norm_num_solutions + w3 * norm_path_length
        normalized_matrix[i, j] = dist
        normalized_matrix[j, i] = dist
    return normalized_matrix
