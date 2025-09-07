import logging
from typing import List, Tuple, Dict, Optional, Set

from brancharchitect.tree import Node
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.elements.partition import Partition
from brancharchitect.jumping_taxa.lattice.lattice_edge import LatticeEdge
from brancharchitect.jumping_taxa.lattice.depth_computation import (
    compute_lattice_edge_depths,
)

# It's good practice to use the logging module
jt_logger: logging.Logger = logging.getLogger(__name__)


def map_transient_sedges_to_original(
    s_edges_from_iteration: List[Partition],
    original_t1: Node,
    original_t2: Node,
) -> List[Optional[Partition]]:
    """
    Maps s-edges from a pruned tree back to the original common splits
    using Jaccard similarity, without caching.

    For each s-edge, it finds the original common split with the highest
    Jaccard index. Ties are broken by choosing the smaller partition.

    Args:
        s_edges_from_iteration: List of s-edge Partitions from an iteration.
        original_t1: The initial, unmodified first tree.
        original_t2: The initial, unmodified second tree.

    Returns:
        A list of the best-matching original Partitions.
    """
    original_common_splits = list(original_t1.to_splits() & original_t2.to_splits())
    mapped_partitions: List[Optional[Partition]] = []

    if not original_common_splits:
        return [None] * len(s_edges_from_iteration)

    for s_edge in s_edges_from_iteration:
        s_edge_indices = set(s_edge.resolve_to_indices())

        best_match: Optional[Partition] = None
        highest_score: float = -1.0
        smallest_size: int = int(1e9)

        for original_p in original_common_splits:
            original_indices = set(original_p.resolve_to_indices())

            intersection_len = len(s_edge_indices.intersection(original_indices))
            if intersection_len == 0:
                continue

            union_len = len(s_edge_indices) + len(original_indices) - intersection_len
            jaccard_index = intersection_len / union_len

            if jaccard_index > highest_score:
                highest_score = jaccard_index
                best_match = original_p
                smallest_size = len(original_indices)
            elif jaccard_index == highest_score:
                if len(original_indices) < smallest_size:
                    best_match = original_p
                    smallest_size = len(original_indices)

        mapped_partitions.append(best_match)

    return mapped_partitions

# Backwards-compatibility alias
def map_s_edges_by_jaccard_similarity(
    s_edges_from_iteration: List[Partition],
    original_t1: Node,
    original_t2: Node,
) -> List[Optional[Partition]]:
    """Deprecated alias for map_transient_sedges_to_original."""
    return map_transient_sedges_to_original(
        s_edges_from_iteration, original_t1, original_t2
    )


# --- Function 2: NumPy Bitmask (The function to test) ---
def pair_candidates_with_solutions_numpy_bitmask(
    candidate_atoms: Set[Partition], solution_atoms: Set[Partition]
) -> Dict[Partition, Partition]:
    """
    Finds the best overlapping unique atoms for each solution.
    For each solution, finds all unique atoms that overlap with it,
    then returns the one with the highest overlap ratio.
    """
    if not candidate_atoms or not solution_atoms:
        return {}

    # Convert sets to lists
    cand_list: List[Partition] = list(candidate_atoms)
    sol_list: List[Partition] = list(solution_atoms)

    # For each solution, find the best overlapping unique atom
    result_mapping: Dict[Partition, Partition] = {}

    for solution in sol_list:
        best_candidate = None
        best_score = 0

        for candidate in cand_list:
            # Calculate overlap (intersection) between candidate and solution bitmasks
            overlap_bits = candidate.bitmask & solution.bitmask
            overlap_count = bin(overlap_bits).count("1") if overlap_bits else 0

            if overlap_count > 0:
                # Score based on overlap ratio - higher is better
                candidate_size = bin(candidate.bitmask).count("1")
                solution_size = bin(solution.bitmask).count("1")

                # Prefer candidates with higher overlap ratio to the solution
                overlap_ratio = overlap_count / solution_size

                # Also consider how much of the candidate overlaps
                candidate_overlap_ratio = overlap_count / candidate_size

                # Combined score favoring good overlap from both perspectives
                score = overlap_ratio + candidate_overlap_ratio

                if score > best_score:
                    best_score = score
                    best_candidate = candidate

        # Map the best overlapping candidate to this solution
        if best_candidate is not None:
            result_mapping[best_candidate] = solution

    return result_mapping


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


def map_solution_elements_to_atoms(
    s_edge_solutions: Dict[Partition, List[List[Partition]]],
    unique_splits_t1: PartitionSet[Partition],
    unique_splits_t2: PartitionSet[Partition],
) -> Tuple[Dict[Partition, Partition], Dict[Partition, Partition]]:
    """
    Map solution elements to their best matching unique atoms.

    For each solution element:
    - If solution is bigger than all unique atoms: use the s_edge
    - Otherwise: use overlap-based mapping to find best matching atom

    Args:
        s_edge_solutions: Dictionary mapping s_edges to their solution sets
        unique_splits_t1: Unique splits for tree 1
        unique_splits_t2: Unique splits for tree 2

    Returns:
        Tuple of (tree1_mapping, tree2_mapping) where each mapping is
        Dict[solution_element, best_matching_atom]
    """
    # Extract unique atoms from both trees
    unique_atoms_t1: PartitionSet[Partition] = unique_splits_t1.atom()
    unique_atoms_t2: PartitionSet[Partition] = unique_splits_t2.atom()

    # Add edges to unique atom sets
    for edge in s_edge_solutions.keys():
        unique_atoms_t1.add(edge)
        unique_atoms_t2.add(edge)

    # Get maximum sizes of original unique atoms (excluding edges)
    original_atoms_t1: PartitionSet[Partition] = unique_splits_t1.atom()
    original_atoms_t2: PartitionSet[Partition] = unique_splits_t2.atom()
    max_atom_size_t1: int = _get_max_atom_size(original_atoms_t1)
    max_atom_size_t2: int = _get_max_atom_size(original_atoms_t2)

    # Initialize result mappings
    solution_to_atom_mapping_t1: Dict[Partition, Partition] = {}
    solution_to_atom_mapping_t2: Dict[Partition, Partition] = {}

    # Process each edge and its solutions
    for edge, edge_solutions in s_edge_solutions.items():
        for solution_set in edge_solutions:
            for solution_element in solution_set:
                solution_size = _get_partition_size(solution_element)

                # Determine mapping for tree 1
                if _should_use_edge_mapping(solution_size, max_atom_size_t1):
                    solution_to_atom_mapping_t1[solution_element] = edge
                else:
                    # Use overlap-based mapping
                    single_solution = {solution_element}
                    mapping_result = pair_candidates_with_solutions_numpy_bitmask(
                        set(unique_atoms_t1), single_solution
                    )
                    for atom, solution in mapping_result.items():
                        solution_to_atom_mapping_t1[solution] = atom

                # Determine mapping for tree 2
                if _should_use_edge_mapping(solution_size, max_atom_size_t2):
                    solution_to_atom_mapping_t2[solution_element] = edge
                else:
                    # Use overlap-based mapping
                    single_solution = {solution_element}
                    mapping_result = pair_candidates_with_solutions_numpy_bitmask(
                        set(unique_atoms_t2), single_solution
                    )
                    for atom, solution in mapping_result.items():
                        solution_to_atom_mapping_t2[solution] = atom

    return solution_to_atom_mapping_t1, solution_to_atom_mapping_t2


# Backwards-compatibility alias
def map_solutions_to_atoms(
    s_edge_solutions: Dict[Partition, List[List[Partition]]],
    unique_splits_t1: PartitionSet[Partition],
    unique_splits_t2: PartitionSet[Partition],
) -> Tuple[Dict[Partition, Partition], Dict[Partition, Partition]]:
    """Deprecated alias for map_solution_elements_to_atoms."""
    return map_solution_elements_to_atoms(
        s_edge_solutions, unique_splits_t1, unique_splits_t2
    )


def sort_lattice_edges_by_subset_hierarchy(
    lattice_edges: List[LatticeEdge], tree1: Node, tree2: Node
) -> List[LatticeEdge]:
    """
    Sort lattice edges by subset hierarchy and tree depth, ensuring smaller sets are processed first.

    Uses the same ordering logic as tree interpolation depth calculation:
    - Subsets are processed before their supersets (e.g., {A} before {A,B})
    - Tree depth provides fine-grained ordering within same subset level
    - Smaller partitions are processed before larger ones

    Args:
        lattice_edges: List of LatticeEdge objects to sort
        tree1: First tree for depth calculation
        tree2: Second tree for depth calculation

    Returns:
        Sorted list of LatticeEdge objects in ascending subset order
    """
    if not lattice_edges:
        return lattice_edges

    # Extract partitions from lattice edges
    partitions = [edge.split for edge in lattice_edges]

    # Compute depths using both trees (use average)
    depths1: Dict[Partition, float] = compute_lattice_edge_depths(partitions, tree1)
    depths2: Dict[Partition, float] = compute_lattice_edge_depths(partitions, tree2)

    # Calculate average depths for sorting
    avg_depths: dict[Partition, float] = {}
    for partition in partitions:
        avg_depths[partition] = (depths1[partition] + depths2[partition]) / 2

    # Sort lattice edges by their average depths (ascending = subsets first)
    sorted_edges: List[LatticeEdge] = sorted(
        lattice_edges,
        key=lambda edge: avg_depths[edge.split],  # type: ignore
    )

    jt_logger.debug(
        f"Sorted {len(lattice_edges)} lattice edges by subset hierarchy and depth"
    )
    depth: float = 0.0
    for i, edge in enumerate(sorted_edges):
        depth = avg_depths[edge.split]
        jt_logger.debug(f"  {i + 1}. {edge.split} (avg_depth={depth})")

    return sorted_edges
