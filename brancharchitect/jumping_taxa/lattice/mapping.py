from typing import List, Tuple, Dict, Optional, Set, FrozenSet
import logging

# Assuming these imports point to valid modules in your project structure
from brancharchitect.tree import Node
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.elements.partition import Partition
from brancharchitect.jumping_taxa.lattice.lattice_edge import LatticeEdge

# It's good practice to use the logging module
jt_logger: logging.Logger = logging.getLogger(__name__)

# --- Caching Helpers for Performance ---
# These would typically be defined in a utility module.
# Using a simple dictionary for demonstration.
_indices_cache: Dict[int, Tuple[int, ...]] = {}
_set_cache: Dict[Tuple[int, ...], FrozenSet[int]] = {}


def _cached_resolve_to_indices(partition: Partition) -> Tuple[int, ...]:
    """Resolves partition to indices, with caching."""
    # Use partition's hash or another unique identifier if available
    cache_key = hash(partition)
    if cache_key not in _indices_cache:
        _indices_cache[cache_key] = partition.resolve_to_indices()
    return _indices_cache[cache_key]


def _cached_set_from_indices(indices: Tuple[int, ...]) -> FrozenSet[int]:
    """Converts a tuple of indices to a frozenset, with caching."""
    if indices not in _set_cache:
        _set_cache[indices] = frozenset(indices)
    return _set_cache[indices]


# --- Main Mapping Logic ---


def _get_mapping_context(
    s_edge: Partition,
    t1: Node,
    t2: Node,
    t1_splits: PartitionSet[Partition],
    t2_splits: PartitionSet[Partition],
) -> Optional[Node]:
    """
    Determines which tree (t1 or t2) the s_edge belongs to in the current iteration.
    """
    if s_edge in t1_splits:
        return t1
    elif s_edge in t2_splits:
        return t2
    return None


def map_s_edges_to_original_by_index(
    s_edges_from_iteration: List[Partition],
    original_t1: Node,
    original_t2: Node,
    current_t1: Node,
    current_t2: Node,
    iteration_count: int,
) -> List[Optional[Partition]]:
    """
    [CORRECTED] Maps s-edge partitions found in a specific iteration back to their
    corresponding partitions from the set of splits *common* to the original trees.

    This ensures that an s-edge is only mapped back to a valid, common ancestral partition.

    Args:
        s_edges_from_iteration: The list of s-edge Partitions from the lattice algorithm.
        original_t1: The initial, unmodified first tree.
        original_t2: The initial, unmodified second tree.
        current_t1: The first tree in the current iteration (potentially pruned).
        current_t2: The second tree in the current iteration (potentially pruned).
        iteration_count: The current iteration number, for logging.

    Returns:
        A list of original Partitions that correspond to the s-edges found
        in the current iteration.
    """

    # Compute the set of splits common to both original trees
    original_common_splits = original_t1.to_splits() & original_t2.to_splits()

    # Only map s-edges that are also present in the original common splits
    mapped_partitions: List[Optional[Partition]] = []
    t1_splits: PartitionSet[Partition] = current_t1.to_splits(with_leaves=True)
    t2_splits: PartitionSet[Partition] = current_t2.to_splits(with_leaves=True)
    t1_leaf_indices: FrozenSet[int] = _cached_set_from_indices(
        _cached_resolve_to_indices(current_t1.split_indices)
    )
    t2_leaf_indices: FrozenSet[int] = _cached_set_from_indices(
        _cached_resolve_to_indices(current_t2.split_indices)
    )

    # Filter s_edges to only those present in original_common_splits
    filtered_s_edges = [
        s for s in s_edges_from_iteration if s in original_common_splits
    ]

    for s_edge in filtered_s_edges:
        context_tree = _get_mapping_context(
            s_edge, current_t1, current_t2, t1_splits, t2_splits
        )

        if not context_tree:
            jt_logger.warning(
                f"Iter {iteration_count}: s_edge {s_edge} not in current t1 or t2 splits. Cannot map."
            )
            mapped_partitions.append(None)
            continue

        context_leaf_indices = (
            t1_leaf_indices if context_tree is current_t1 else t2_leaf_indices
        )
        s_edge_indices_set = _cached_set_from_indices(
            _cached_resolve_to_indices(s_edge)
        )

        # Diagnostic logging for mapping failures
        jt_logger.debug(
            f"Iter {iteration_count}: s_edge indices: {sorted(s_edge_indices_set)}"
        )
        jt_logger.debug(
            f"Iter {iteration_count}: context leaf indices: {sorted(context_leaf_indices)}"
        )
        candidate_found = False
        for original_p in original_common_splits:
            original_indices_set = _cached_set_from_indices(
                _cached_resolve_to_indices(original_p)
            )
            intersection = original_indices_set & context_leaf_indices
            jt_logger.debug(
                f"Iter {iteration_count}: Trying original_p {original_p}, indices: {sorted(original_indices_set)}, intersection: {sorted(intersection)}"
            )
            if s_edge_indices_set == intersection:
                candidate_found = True
                if original_p not in mapped_partitions:
                    mapped_partitions.append(original_p)
                else:
                    mapped_partitions.append(None)
                break
        if not candidate_found:
            jt_logger.warning(
                f"Iter {iteration_count}: No match found for s_edge {s_edge}. s_edge_indices: {sorted(s_edge_indices_set)}, context_leaf_indices: {sorted(context_leaf_indices)}"
            )
            mapped_partitions.append(None)

    return mapped_partitions


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
    result_mapping = {}

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


def map_solutions_to_atoms(
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

    # Import here to avoid circular import
    from brancharchitect.tree_interpolation.ordering import compute_lattice_edge_depths

    # Extract partitions from lattice edges
    partitions = [edge.split for edge in lattice_edges]

    # Compute depths using both trees (use average)
    depths1: Dict[Partition, float] = compute_lattice_edge_depths(partitions, tree1)
    depths2: Dict[Partition, float] = compute_lattice_edge_depths(partitions, tree2)

    # Calculate average depths for sorting
    avg_depths = {}
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
    for i, edge in enumerate(sorted_edges):
        depth: float = avg_depths[edge.split]
        jt_logger.debug(f"  {i + 1}. {edge.split} (avg_depth={depth})")

    return sorted_edges
