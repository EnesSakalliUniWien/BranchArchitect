import random
import itertools
from tqdm import tqdm
from copy import deepcopy
from math import factorial
from collections import Counter
from brancharchitect.tree import Node
from typing import Tuple, List, Any
from brancharchitect.leaforder.circular_distances import (
    circular_distance_tree_pair,
    circular_distances_trees,
)

# --- Utilities ---
def flatten_tree_list(tree_list: Any) -> List[str]:
    """Flatten a nested tree list into a linear list of taxa names."""
    if isinstance(tree_list, list):
        flattened = []
        for element in tree_list:
            flattened.extend(flatten_tree_list(element))
        return flattened
    else:
        return [tree_list]


def generate_permutations(
    elements, fixed_positions=None, fixed_elements=None, sample_size=None, seed=None
):
    """
    Generate unique random permutations of the elements list, while keeping specified elements fixed.
    If the total search space is small or we need all permutations, we may directly enumerate them.
    Otherwise, we attempt random sampling.

    Args:
        elements (list): The list of elements to permute.
        fixed_positions (list): Indices of positions to keep fixed. These positions won't change.
        fixed_elements (list): Elements to keep fixed (all their occurrences remain in place).
        sample_size (int): Number of unique permutations to generate. If None, we generate all unique permutations.
        seed (int): Seed for random number generator (for reproducibility).

    Returns:
        list: A list of unique permutations (each permutation is a list).
    """
    if seed is not None:
        random.seed(seed)

    n = len(elements)
    positions = list(range(n))

    # Validate and prepare fixed_positions
    if fixed_positions is None:
        fixed_positions = []
    else:
        # Ensure positions are within range
        fixed_positions = [pos for pos in fixed_positions if 0 <= pos < n]

    # Identify fixed positions based on fixed_elements if provided
    if fixed_elements:
        # Add positions of elements that must be fixed
        for i, e in enumerate(elements):
            if e in fixed_elements:
                fixed_positions.append(i)
        # Deduplicate
        fixed_positions = list(set(fixed_positions))

    # Determine which elements to permute
    fixed_positions_set = set(fixed_positions)
    permute_positions = [pos for pos in positions if pos not in fixed_positions_set]
    permute_elements = [elements[pos] for pos in permute_positions]

    # Calculate the total number of unique permutations of permute_elements
    element_counts = Counter(permute_elements)
    total_unique_permutations = factorial(len(permute_elements))
    for count in element_counts.values():
        total_unique_permutations //= factorial(count)

    # If sample_size is None or larger than total_unique_permutations, adjust it
    if sample_size is None or sample_size > total_unique_permutations:
        sample_size = total_unique_permutations

    # If total_unique_permutations is small and manageable, or we want all permutations, generate them directly
    # Direct enumeration can be expensive if factorial is large, so we limit when to do this.
    # For example, if total_unique_permutations < 50000 (arbitrary limit), we do direct enumeration.
    # Adjust as needed.
    DIRECT_ENUMERATION_LIMIT = 50000
    if total_unique_permutations <= DIRECT_ENUMERATION_LIMIT:
        # Generate all unique permutations of permute_elements
        all_perms_set = set(itertools.permutations(permute_elements))
        all_perms_list = list(all_perms_set)

        # If sample_size < total_unique_permutations, randomly sample from them
        if sample_size < total_unique_permutations:
            selected = random.sample(all_perms_list, sample_size)
        else:
            selected = all_perms_list  # Return all permutations

        # Reinsert fixed elements into their positions
        result = []
        for perm in selected:
            perm = list(perm)
            full = elements.copy()
            # Place permute_elements back
            for idx, pos in enumerate(permute_positions):
                full[pos] = perm[idx]
            # fixed_positions remain unchanged
            result.append(full)
        return result

    # Otherwise, try random attempts:
    generated_permutations = set()
    attempts = 0
    max_attempts = sample_size * 10  # to avoid infinite loops

    while len(generated_permutations) < sample_size and attempts < max_attempts:
        # Generate a random permutation of the permute elements
        permuted_elements = permute_elements[:]
        random.shuffle(permuted_elements)
        # Reconstruct the full permutation by placing fixed elements back
        temp = elements.copy()
        for idx, pos in enumerate(permute_positions):
            temp[pos] = permuted_elements[idx]
        perm_tuple = tuple(temp)
        if perm_tuple not in generated_permutations:
            generated_permutations.add(perm_tuple)
        attempts += 1

    if len(generated_permutations) < sample_size:
        print(
            f"Warning: Could only generate {len(generated_permutations)} unique permutations "
            f"out of requested {sample_size} after {attempts} attempts."
        )

    if attempts >= max_attempts:
        print(
            f"Reached maximum number of attempts ({max_attempts}). Returning {len(generated_permutations)} unique permutations."
        )

    return [list(perm) for perm in generated_permutations]


def find_minimal_distance_permutation(trees: List, permutations: List):
    # Find the permutation that minimizes the total circular distance
    min_total_distance = float("inf")
    best_perm = None
    for perm in tqdm(permutations, desc="Finding global optimal order"):
        # Apply permutation to all trees
        for tree in trees:
            tree.reorder_taxa(perm)
        # Compute total distance
        total_distance = circular_distances_trees(trees)
        if total_distance < min_total_distance:
            min_total_distance = total_distance
            best_perm = perm.copy()
    return best_perm


### Split Based Approach ####
def get_taxa_order_in_subtree(node: Node) -> List[str]:
    """Get the taxa order in the subtree rooted at the given node."""
    leaves = node.get_leaves()
    return [leaf.name for leaf in leaves]


def restore_original_orders(trees: List, original_orders: List):
    # Restore original orders
    for tree, original_order in zip(trees, original_orders):
        tree.reorder_taxa(original_order)


def optimize_tree_rotation(consensus_tree, target_trees: List, n_iterations: int = 100):
    """
    Optimize the consensus tree by rotating subtrees (swapping children at internal nodes)
    to minimize total distance to target trees.

    Improvements:
    - Search all possible pairs of children in each node before deciding on the best improvement.
    - Only commit to the single best improvement per node per iteration.
    - If no improvements are found in an iteration, stop early.
    """

    def compare_tree_with_target_trees(c_tree, t_trees):
        return sum(
            circular_distance_tree_pair(c_tree, target_tree) for target_tree in t_trees
        )

    best_tree = deepcopy(consensus_tree)
    total_distances = []
    best_distance = compare_tree_with_target_trees(best_tree, target_trees)
    total_distances.append(best_distance)

    for iteration in range(n_iterations):
        improved = False

        # Identify internal nodes that can be permuted (2 or more children)
        internal_nodes = [
            node
            for node in best_tree.traverse()
            if node.children and len(node.children) >= 2
        ]

        # Try improving each node by finding the best swap
        for node in internal_nodes:
            original_children = node.children[:]
            node_best_children = original_children
            node_best_distance = best_distance
            found_improvement_for_node = False

            num_children = len(node.children)
            # Evaluate all pairwise swaps to find the best improvement
            for i in range(num_children):
                for j in range(i + 1, num_children):
                    # Swap children i and j
                    node.children[i], node.children[j] = (
                        node.children[j],
                        node.children[i],
                    )

                    current_distance = compare_tree_with_target_trees(
                        best_tree, target_trees
                    )
                    if current_distance < node_best_distance:
                        # Found a better configuration for this node
                        node_best_distance = current_distance
                        node_best_children = node.children[:]
                        found_improvement_for_node = True

                    # Revert to continue searching
                    node.children[i], node.children[j] = (
                        node.children[j],
                        node.children[i],
                    )

            # After checking all pairs, apply the best improvement if found
            if found_improvement_for_node and node_best_distance < best_distance:
                node.children = node_best_children
                best_distance = node_best_distance
                total_distances.append(best_distance)
                improved = True
            else:
                # No improvement for this node, revert changes
                node.children = original_children

        # If no improvements in this entire iteration, stop
        if not improved:
            break

    return best_tree, total_distances


def collect_distances_for_trajectory(
    trees: List[Node],
) -> Tuple[List[float], List[Tuple[int, int]]]:
    """
    Collect pairwise distances for a sequence of trees, including the indices of tree pairs.

    Args:
        trees (List[Node]): List of trees.

    Returns:
        Tuple[List[float], List[Tuple[int, int]]]: Distances and corresponding tree pairs.
    """
    distances = []
    pairs = []
    for i in range(len(trees) - 1):
        distance = circular_distance_tree_pair(trees[i], trees[i + 1])
        distances.append(distance)
        pairs.append((i, i + 1))
    return distances, pairs


def global_consensus_smoothing(trees: List["Node"], max_samples: int = 1000) -> None:
    """
    Attempt to find a global ordering of taxa that minimizes total circular distance
    across all trees, and reorder all trees to that global ordering.

    Args:
        trees (List["Node"]): The list of trees to smooth.
        max_samples (int): Number of permutations to sample if exhaustive search is not feasible.
    """

    # Extract the set of all taxa
    taxa = list({leaf.name for tree in trees for leaf in tree.get_leaves()})
    original_orders = [tree.get_current_order() for tree in trees]

    # Generate random permutations or use heuristics to find a near-optimal global order
    # For simplicity, let's just generate random permutations and pick the best
    # If the number of taxa is large, consider a smarter heuristic or reduce max_samples
    permutations = generate_permutations(taxa, sample_size=min(max_samples, 1000))

    best_perm = None
    best_distance = float("inf")

    for perm in permutations:
        # Apply this permutation to all trees
        for tree in trees:
            tree.reorder_taxa(perm)
        # Compute total circular distance across the entire trajectory
        total_distance = sum(
            circular_distance_tree_pair(trees[i], trees[i + 1])
            for i in range(len(trees) - 1)
        )

        if total_distance < best_distance:
            best_distance = total_distance
            best_perm = perm

    # Reorder all trees to the best permutation found
    if best_perm is not None:
        for tree in trees:
            tree.reorder_taxa(best_perm)
    else:
        # If no improvement found, revert to original orders
        for tree, order in zip(trees, original_orders):
            tree.reorder_taxa(list(order))
