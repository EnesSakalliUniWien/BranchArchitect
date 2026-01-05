from typing import Iterable, Set, cast, TypeVar
from brancharchitect.elements.partition import Partition

T = TypeVar("T", bound="Partition")


def solve_minimum_set_cover(partitions: Iterable[T]) -> Set[T]:
    """
    Compute a minimum-cardinality union cover (minimum set cover) of the given partitions.

    Finds a subset C ⊆ partitions with the fewest elements such that
    union(C) == union(partitions). This uses an exact branch-and-bound search with
    greedy lower-bound pruning. It is deterministic for a fixed input.

    Args:
        partitions: An iterable of Partition objects.

    Returns:
        Set[T]: A set containing the partitions that form the minimum cover.
    """
    # --- 1. Preparation ---

    # Filter out empty partitions as they cannot contribute to coverage
    valid_partitions: list[Partition] = [p for p in partitions if len(p.indices) > 0]

    # Identify the "Universe": the total set of indices that must be covered
    universe: set[int] = set()
    for p in valid_partitions:
        universe.update(p.indices)

    if not universe:
        return set()

    num_partitions = len(valid_partitions)

    # Sort partitions by size (descending) to prioritize larger sets.
    # This helps the greedy heuristic and allows the DFS to find small covers faster.
    # Secondary sort by bitmask ensures deterministic results.
    valid_partitions.sort(key=lambda p: (-len(p.indices), p.bitmask))

    # Cache the indices of each partition as sets for efficient set operations
    partition_indices_sets = [set(p.indices) for p in valid_partitions]

    # --- 2. Precomputation for Pruning ---

    # Precompute "Suffix Unions": suffix_union[i] is the union of all partitions from index i onwards.
    # Used to check if the remaining universe can possibly be covered by remaining partitions.
    suffix_union: list[set[int]] = [set() for _ in range(num_partitions + 1)]
    for i in range(num_partitions - 1, -1, -1):
        suffix_union[i] = set(suffix_union[i + 1])
        suffix_union[i].update(partition_indices_sets[i])

    # --- 3. Initial Upper Bound (Greedy Heuristic) ---

    # Run a greedy algorithm to find an initial valid solution.
    # This gives us a "best_size" to start pruning against immediately.
    greedy_solution_indices: list[int] = []
    greedy_remaining_universe = set(universe)
    available_indices = list(range(num_partitions))

    while greedy_remaining_universe and available_indices:
        best_idx = -1
        max_gain = 0

        # Find the partition that covers the most *uncovered* elements
        for idx in available_indices:
            gain = len(partition_indices_sets[idx] & greedy_remaining_universe)
            if gain > max_gain:
                max_gain = gain
                best_idx = idx

        if best_idx == -1:
            break  # Should not happen if universe is coverable

        greedy_solution_indices.append(best_idx)
        greedy_remaining_universe -= partition_indices_sets[best_idx]
        available_indices.remove(best_idx)

    # Initialize best solution found so far
    if not greedy_remaining_universe:
        best_solution_indices = greedy_solution_indices
        best_solution_size = len(best_solution_indices)
    else:
        # Fallback (should be unreachable given logic above)
        best_solution_size = num_partitions + 1
        best_solution_indices = []

    # --- 4. Exact Search (Branch and Bound DFS) ---

    def dfs(
        current_idx: int, current_solution: list[int], current_covered: set[int]
    ) -> None:
        nonlocal best_solution_size, best_solution_indices

        # Pruning 1: Current path is already longer than or equal to the best found solution.
        if len(current_solution) >= best_solution_size:
            return

        remaining_to_cover = universe - current_covered

        # Base Case: Universe is fully covered
        if not remaining_to_cover:
            best_solution_size = len(current_solution)
            best_solution_indices = list(current_solution)
            return

        # Base Case: No more partitions to consider
        if current_idx >= num_partitions:
            return

        # Pruning 2 (Feasibility): Impossible to cover remaining elements with remaining partitions.
        if not remaining_to_cover.issubset(suffix_union[current_idx]):
            return

        # Pruning 3 (Lower Bound / Admissible Heuristic):
        # Estimate minimum sets needed. Since partitions are sorted by size,
        # the largest remaining set is partition_indices_sets[current_idx].
        # We need at least ceil(|remaining| / |largest_next_set|) more sets.
        largest_available_size = len(partition_indices_sets[current_idx])
        min_sets_needed = (
            len(remaining_to_cover) + largest_available_size - 1
        ) // largest_available_size

        if len(current_solution) + min_sets_needed >= best_solution_size:
            return

        # Branch A: Include the current partition
        # Only useful if it covers new elements
        newly_covered = partition_indices_sets[current_idx] - current_covered
        if newly_covered:
            current_solution.append(current_idx)
            dfs(
                current_idx + 1,
                current_solution,
                current_covered | partition_indices_sets[current_idx],
            )
            current_solution.pop()  # Backtrack

        # Branch B: Exclude the current partition
        dfs(current_idx + 1, current_solution, current_covered)

    # Start the search
    dfs(0, [], set())

    return {cast(T, valid_partitions[i]) for i in best_solution_indices}
