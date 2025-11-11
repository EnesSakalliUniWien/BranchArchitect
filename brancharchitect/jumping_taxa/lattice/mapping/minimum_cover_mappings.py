"""
Minimum cover mapping utilities for lattice algorithm.

Maps solution elements to partitions from the minimum covers of unique splits
for each tree under a pivot.
"""

from typing import Dict, List, Tuple, Optional, Set

from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet


def find_best_overlapping_partition(
    candidate_partitions: Set[Partition], solution: Partition
) -> Optional[Partition]:
    if not candidate_partitions:
        return None
    best_partition = None
    max_overlap = 0
    best_partition_size = float("inf")
    for partition in candidate_partitions:
        overlap_bits = partition.bitmask & solution.bitmask
        overlap_count = bin(overlap_bits).count("1") if overlap_bits else 0
        if overlap_count > 0:
            partition_size = bin(partition.bitmask).count("1")
            if (overlap_count > max_overlap) or (
                overlap_count == max_overlap and partition_size < best_partition_size
            ):
                max_overlap = overlap_count
                best_partition = partition
                best_partition_size = partition_size
    return best_partition


def _get_partition_size(partition: Partition) -> int:
    return bin(partition.bitmask).count("1")


def _get_max_partition_size(partitions: PartitionSet[Partition]) -> int:
    return max((_get_partition_size(partition) for partition in partitions), default=0)


def _should_use_edge_mapping(solution_size: int, max_partition_size: int) -> bool:
    return solution_size > max_partition_size


def _map_solution_to_partition(
    solution_element: Partition,
    edge: Partition,
    min_cover_partitions: PartitionSet[Partition],
    max_partition_size: int,
) -> Partition:
    solution_size = _get_partition_size(solution_element)
    if _should_use_edge_mapping(solution_size, max_partition_size):
        return edge
    best_partition = find_best_overlapping_partition(set(min_cover_partitions), solution_element)
    return best_partition if best_partition else edge


def map_solution_elements_to_minimum_covers(
    pivot_edge_solutions: Dict[Partition, List[Partition]],
    unique_splits_t1: PartitionSet[Partition],
    unique_splits_t2: PartitionSet[Partition],
) -> Tuple[
    Dict[Partition, Dict[Partition, Partition]],
    Dict[Partition, Dict[Partition, Partition]],
]:
    min_cover_t1: PartitionSet[Partition] = unique_splits_t1.minimum_cover()
    min_cover_t2: PartitionSet[Partition] = unique_splits_t2.minimum_cover()

    max_cover_size_t1: int = _get_max_partition_size(min_cover_t1)
    max_cover_size_t2: int = _get_max_partition_size(min_cover_t2)

    active_changing_splits_atom_map_one: Dict[Partition, Dict[Partition, Partition]] = {}
    active_changing_splits_atom_map_two: Dict[Partition, Dict[Partition, Partition]] = {}

    for edge, edge_partitions in pivot_edge_solutions.items():
        active_changing_splits_atom_map_one[edge] = {}
        active_changing_splits_atom_map_two[edge] = {}

        current_min_cover_t1 = min_cover_t1 | {edge}
        current_min_cover_t2 = min_cover_t2 | {edge}

        for solution_element in edge_partitions:
            mapped_partition_t1 = _map_solution_to_partition(
                solution_element, edge, current_min_cover_t1, max_cover_size_t1
            )
            active_changing_splits_atom_map_one[edge][solution_element] = mapped_partition_t1

            mapped_partition_t2 = _map_solution_to_partition(
                solution_element, edge, current_min_cover_t2, max_cover_size_t2
            )
            active_changing_splits_atom_map_two[edge][solution_element] = mapped_partition_t2

    return active_changing_splits_atom_map_one, active_changing_splits_atom_map_two


def map_solution_elements_to_minimal_frontiers(
    pivot_edge_solutions: Dict[Partition, List[Partition]],
    unique_splits_t1: PartitionSet[Partition],
    unique_splits_t2: PartitionSet[Partition],
) -> Tuple[
    Dict[Partition, Dict[Partition, Partition]],
    Dict[Partition, Dict[Partition, Partition]],
]:
    """
    Map solution elements to pivot-local minimal unique splits ("minimal frontiers").

    For each pivot edge, we compute the minimal elements of the unique splits under
    that pivot for t1 and t2, and map every solution element to the best-overlapping
    partition among these minimal frontiers (with the pivot as fallback).

    Returns two dictionaries (for t1 and t2) keyed by pivot edge, each mapping a
    solution partition -> selected minimal-frontier partition (or the pivot edge).
    """
    # Compute minimal unique frontiers under the pivot for both trees
    min_frontier_t1: PartitionSet[Partition] = unique_splits_t1.minimal_elements()
    min_frontier_t2: PartitionSet[Partition] = unique_splits_t2.minimal_elements()

    max_size_t1: int = _get_max_partition_size(min_frontier_t1)
    max_size_t2: int = _get_max_partition_size(min_frontier_t2)

    mapped_one: Dict[Partition, Dict[Partition, Partition]] = {}
    mapped_two: Dict[Partition, Dict[Partition, Partition]] = {}

    for edge, edge_partitions in pivot_edge_solutions.items():
        mapped_one[edge] = {}
        mapped_two[edge] = {}

        # Include pivot edge as candidate to ensure a valid mapping fallback
        candidates_t1 = min_frontier_t1 | {edge}
        candidates_t2 = min_frontier_t2 | {edge}

        for solution_element in edge_partitions:
            # Map to t1 frontier
            mapped_partition_t1 = _map_solution_to_partition(
                solution_element, edge, candidates_t1, max_size_t1
            )
            mapped_one[edge][solution_element] = mapped_partition_t1

            # Map to t2 frontier
            mapped_partition_t2 = _map_solution_to_partition(
                solution_element, edge, candidates_t2, max_size_t2
            )
            mapped_two[edge][solution_element] = mapped_partition_t2

    return mapped_one, mapped_two
