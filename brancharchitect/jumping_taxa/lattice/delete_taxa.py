from brancharchitect.tree import Node
from brancharchitect.jumping_taxa.debug import (
    jt_logger,
)
from typing import List, Set, Tuple  # Added TypingSet, TypeVar, cast
from brancharchitect.elements.partition import Partition


def identify_and_delete_jumping_taxa(
    solution_elements_this_iter: List[Partition],
    current_t1: Node,
    current_t2: Node,
    iteration_count: int,
) -> Tuple[bool, Set[int]]:
    """
    Identify jumping taxa from solutions and delete them from trees.

    Args:
        solution_elements_this_iter: Flat list of solution partitions from this iteration
        current_t1: Current state of first tree (modified in-place)
        current_t2: Current state of second tree (modified in-place)
        iteration_count: Current iteration number

    Returns:
        Tuple containing:
        - should_break_loop: Boolean indicating if loop should break
        - taxa_indices_to_delete: Set of taxa indices that were deleted
    """
    # Collect taxa indices from all solution partitions
    taxa_indices_to_delete: Set[int] = set()

    # Debug logging
    jt_logger.info(
        f"[delete_taxa] solution_elements_this_iter type: {type(solution_elements_this_iter)}"
    )
    jt_logger.info(
        f"[delete_taxa] solution_elements_this_iter length: {len(solution_elements_this_iter)}"
    )
    jt_logger.info(
        f"[delete_taxa] solution_elements_this_iter content: {solution_elements_this_iter}"
    )

    for i, sol_partition in enumerate(solution_elements_this_iter):
        jt_logger.info(
            f"[delete_taxa] partition {i} type: {type(sol_partition)}"
        )
        jt_logger.info(f"[delete_taxa] partition {i}: {sol_partition}")
        indices = sol_partition.resolve_to_indices()
        jt_logger.info(f"[delete_taxa] partition {i} indices: {indices}")
        taxa_indices_to_delete.update(indices)

    # Check if there are any taxa to delete
    if not taxa_indices_to_delete:
        jt_logger.warning(
            f"Iter {iteration_count}: Solutions found, but no taxa indices to delete. Breaking."
        )
        return True, taxa_indices_to_delete  # Break loop

    # Log and perform deletion
    jt_logger.info(
        f"Iter {iteration_count}: Deleting taxa indices: {solution_elements_this_iter}"
    )

    current_t1.delete_taxa(list(taxa_indices_to_delete))
    current_t2.delete_taxa(list(taxa_indices_to_delete))

    jt_logger.debug(
        f"Iter {iteration_count}: Deleted taxa indices: {taxa_indices_to_delete}"
    )
    jt_logger.debug(
        f"Iter {iteration_count}: Trees now have {current_t1.to_newick(lengths=False)} and {current_t2.to_newick(lengths=False)} leaves."
    )

    # Check if trees have enough leaves to continue
    if len(current_t1.get_leaves()) < 2 or len(current_t2.get_leaves()) < 2:
        jt_logger.info(
            f"Iter {iteration_count}: One or both trees have too few leaves. Stopping."
        )
        return True, taxa_indices_to_delete  # Break loop

    return False, taxa_indices_to_delete  # Continue loop
