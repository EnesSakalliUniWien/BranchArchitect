import random
import itertools
from math import factorial
from collections import Counter
from brancharchitect.tree import Node
from typing import Tuple, List, Any, Optional, Set
from brancharchitect.leaforder.circular_distances import (
    circular_distance_tree_pair,
    circular_distances_trees,
)


def generate_permutations(
    elements: List[Any],
    fixed_positions: Optional[List[int]] = None,
    fixed_elements: Optional[List[Any]] = None,
    sample_size: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[List[Any]]:
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

    n: int = len(elements)
    positions: List[int] = list(range(n))

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
    fixed_positions_set: Set[int] = set(fixed_positions)
    permute_positions: List[int] = [
        pos for pos in positions if pos not in fixed_positions_set
    ]
    permute_elements: List[Any] = [elements[pos] for pos in permute_positions]

    # Calculate the total number of unique permutations of permute_elements
    element_counts: Counter[Any] = Counter(permute_elements)
    total_unique_permutations: int = factorial(len(permute_elements))
    for count in element_counts.values():
        total_unique_permutations //= factorial(count)

    # If sample_size is None or larger than total_unique_permutations, adjust it
    if sample_size is None or sample_size > total_unique_permutations:
        sample_size = total_unique_permutations

    # If total_unique_permutations is small and manageable, or we want all permutations, generate them directly
    # Direct enumeration can be expensive if factorial is large, so we limit when to do this.
    # For example, if total_unique_permutations < 50000 (arbitrary limit), we do direct enumeration.
    # Adjust as needed.
    DIRECT_ENUMERATION_LIMIT: int = 50000
    if total_unique_permutations <= DIRECT_ENUMERATION_LIMIT:
        # Generate all unique permutations of permute_elements
        all_perms_set: Set[Tuple[Any, ...]] = set(
            itertools.permutations(permute_elements)
        )
        all_perms_list: List[Tuple[Any, ...]] = list(all_perms_set)

        # If sample_size < total_unique_permutations, randomly sample from them
        selected: List[Tuple[Any, ...]]
        if sample_size < total_unique_permutations:
            selected = random.sample(all_perms_list, sample_size)
        else:
            selected = all_perms_list  # Return all permutations

        # Reinsert fixed elements into their positions
        result: List[List[Any]] = []
        for perm in selected:
            perm_list: List[Any] = list(perm)
            full: List[Any] = elements.copy()
            # Place permute_elements back
            for idx, pos in enumerate(permute_positions):
                full[pos] = perm_list[idx]
            # fixed_positions remain unchanged
            result.append(full)
        return result

    # Otherwise, try random attempts:
    generated_permutations: Set[Tuple[Any, ...]] = set()
    attempts: int = 0
    max_attempts: int = sample_size * 10  # to avoid infinite loops

    while len(generated_permutations) < sample_size and attempts < max_attempts:
        # Generate a random permutation of the permute elements
        permuted_elements: List[Any] = permute_elements[:]
        random.shuffle(permuted_elements)
        # Reconstruct the full permutation by placing fixed elements back
        temp: List[Any] = elements.copy()
        for idx, pos in enumerate(permute_positions):
            temp[pos] = permuted_elements[idx]
        perm_tuple: Tuple[Any, ...] = tuple(temp)
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


def find_minimal_distance_permutation(
    trees: List[Node], permutations: List[List[Any]]
) -> Optional[List[Any]]:
    # Find the permutation that minimizes the total circular distance
    min_total_distance: float = float("inf")
    best_perm: Optional[List[Any]] = None
    for perm in permutations:
        # Apply permutation to all trees
        for tree in trees:
            tree.reorder_taxa(perm)
        # Compute total distance
        total_distance: float = circular_distances_trees(trees)
        if total_distance < min_total_distance:
            min_total_distance = total_distance
            best_perm = perm.copy()
    return best_perm


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
    distances: List[float] = []
    pairs: List[Tuple[int, int]] = []
    for i in range(len(trees) - 1):
        distance: float = circular_distance_tree_pair(trees[i], trees[i + 1])
        distances.append(distance)
        pairs.append((i, i + 1))
    return distances, pairs
