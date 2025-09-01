from brancharchitect.tree import Node
from brancharchitect.jumping_taxa.debug import (
    jt_logger,
)
from typing import List, Set, Tuple  # Added TypingSet, TypeVar, cast
from brancharchitect.elements.partition import Partition


def identify_and_delete_jumping_taxa(
    solution_sets_this_iter: List[List[Partition]],
    current_t1: Node,
    current_t2: Node,
    iteration_count: int,
) -> Tuple[bool, Set[int]]:
    """
    Identify jumping taxa from solutions and delete them from trees.

    Args:
        solution_sets_this_iter: Solution sets from current iteration
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
    for solution_set in solution_sets_this_iter:
        for sol_partition in solution_set:
            taxa_indices_to_delete.update(sol_partition.resolve_to_indices())

    # Check if there are any taxa to delete
    if not taxa_indices_to_delete:
        jt_logger.warning(
            f"Iter {iteration_count}: Solutions found, but no taxa indices to delete. Breaking."
        )
        return True, taxa_indices_to_delete  # Break loop

    # Log and perform deletion
    jt_logger.info(
        f"Iter {iteration_count}: Deleting taxa indices: {list(taxa_indices_to_delete)}"
    )
    current_t1.delete_taxa(list(taxa_indices_to_delete))
    current_t2.delete_taxa(list(taxa_indices_to_delete))

    # Check if trees have enough leaves to continue
    if len(current_t1.get_leaves()) < 2 or len(current_t2.get_leaves()) < 2:
        jt_logger.info(
            f"Iter {iteration_count}: One or both trees have too few leaves. Stopping."
        )
        return True, taxa_indices_to_delete  # Break loop
    return False, taxa_indices_to_delete  # Continue loop
