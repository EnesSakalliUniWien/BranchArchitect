"""Atom mapping utilities for lattice algorithm."""

from typing import Dict, List, Tuple, Optional, Set

from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet


def find_best_overlapping_atom(
    candidate_atoms: Set[Partition], solution: Partition
) -> Optional[Partition]:
    """
    Find the atom with the best overlap for a given solution.
    Prioritizes atoms with maximum overlap, and among those with equal overlap,
    selects the smallest atom (by size).
    """
    if not candidate_atoms:
        return None

    best_atom = None
    max_overlap = 0
    best_atom_size = float("inf")

    for atom in candidate_atoms:
        # Calculate overlap between atom and solution
        overlap_bits = atom.bitmask & solution.bitmask
        overlap_count = bin(overlap_bits).count("1") if overlap_bits else 0

        if overlap_count > 0:  # Only consider atoms with some overlap
            atom_size = bin(atom.bitmask).count("1")

            # Select atom if it has better overlap, or same overlap but smaller size
            if (overlap_count > max_overlap) or (
                overlap_count == max_overlap and atom_size < best_atom_size
            ):
                max_overlap = overlap_count
                best_atom = atom
                best_atom_size = atom_size

    return best_atom


# --- Helper Functions for Solution-to-Atom Mapping ---


def _get_partition_size(partition: Partition) -> int:
    """Get the size (number of taxa) in a partition."""
    return bin(partition.bitmask).count("1")


def _get_max_atom_size(atoms: PartitionSet[Partition]) -> int:
    """Get the maximum size of atoms in a partition set."""
    return max((_get_partition_size(atom) for atom in atoms), default=0)


def _should_use_edge_mapping(solution_size: int, max_atom_size: int) -> bool:
    """Determine if we should use edge mapping instead of overlap-based mapping."""
    return solution_size > max_atom_size


def _map_solution_to_atom(
    solution_element: Partition,
    edge: Partition,
    unique_atoms: PartitionSet[Partition],
    max_atom_size: int,
) -> Partition:
    """Map a single solution element to its best matching atom."""
    solution_size = _get_partition_size(solution_element)

    if _should_use_edge_mapping(solution_size, max_atom_size):
        return edge
    else:
        # Use overlap-based mapping to find best atom
        best_atom = find_best_overlapping_atom(set(unique_atoms), solution_element)
        # Fallback to edge if no overlapping atom found
        return best_atom if best_atom else edge


def map_solution_elements_to_atoms(
    s_edge_solutions: Dict[Partition, List[List[Partition]]],
    unique_splits_t1: PartitionSet[Partition],
    unique_splits_t2: PartitionSet[Partition],
) -> Tuple[
    Dict[Partition, Dict[Partition, Partition]],
    Dict[Partition, Dict[Partition, Partition]],
]:
    """
    Map solution elements (per s-edge) to their best matching atoms in each tree.

    For each solution element:
    - If the solution is larger than any candidate atom: map to its s-edge
    - Otherwise: map to the best-overlapping atom

    Args:
        s_edge_solutions: Dictionary mapping s_edges to their solution sets
        unique_splits_t1: Unique splits for tree 1
        unique_splits_t2: Unique splits for tree 2

    Returns:
        Tuple of per-edge mappings:
        - Tree 1: {edge -> {solution_element -> best_atom}}
        - Tree 2: {edge -> {solution_element -> best_atom}}
    """
    # Use a copy of unique splits as candidates to avoid modifying the original
    unique_atoms_t1: PartitionSet[Partition] = unique_splits_t1.copy()
    unique_atoms_t2: PartitionSet[Partition] = unique_splits_t2.copy()

    # Get maximum sizes of original unique splits (excluding edges)
    max_atom_size_t1: int = _get_max_atom_size(unique_atoms_t1.atom())
    max_atom_size_t2: int = _get_max_atom_size(unique_atoms_t2.atom())

    active_changing_splits_atom_map_one: Dict[
        Partition, Dict[Partition, Partition]
    ] = {}
    active_changing_splits_atom_map_two: Dict[
        Partition, Dict[Partition, Partition]
    ] = {}

    # Process each edge and its solutions
    for edge, edge_solutions in s_edge_solutions.items():
        # Initialize per-edge mapping dictionaries
        active_changing_splits_atom_map_one[edge] = {}
        active_changing_splits_atom_map_two[edge] = {}

        # Add edge to atom sets on-the-fly for this iteration
        current_unique_atoms_t1 = unique_atoms_t1 | {edge}
        current_unique_atoms_t2 = unique_atoms_t2 | {edge}

        for solution_set in edge_solutions:
            for solution_element in solution_set:
                # Map for tree 1
                mapped_atom_t1 = _map_solution_to_atom(
                    solution_element, edge, current_unique_atoms_t1, max_atom_size_t1
                )
                active_changing_splits_atom_map_one[edge][solution_element] = (
                    mapped_atom_t1
                )

                # Map for tree 2
                mapped_atom_t2 = _map_solution_to_atom(
                    solution_element, edge, current_unique_atoms_t2, max_atom_size_t2
                )
                active_changing_splits_atom_map_two[edge][solution_element] = (
                    mapped_atom_t2
                )

    return (
        active_changing_splits_atom_map_one,
        active_changing_splits_atom_map_two,
    )


# Backwards-compatibility alias (returns per-edge mappings)
def map_solutions_to_atoms(
    s_edge_solutions: Dict[Partition, List[List[Partition]]],
    unique_splits_t1: PartitionSet[Partition],
    unique_splits_t2: PartitionSet[Partition],
) -> Tuple[
    Dict[Partition, Dict[Partition, Partition]],
    Dict[Partition, Dict[Partition, Partition]],
]:
    """Alias for map_solution_elements_to_atoms returning per-edge mappings."""
    return map_solution_elements_to_atoms(
        s_edge_solutions, unique_splits_t1, unique_splits_t2
    )
