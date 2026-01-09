from typing import Dict, List, Set

from brancharchitect.elements.partition import Partition
from brancharchitect.tree import Node
from brancharchitect.logger import jt_logger


def identify_and_delete_jumping_taxa(
    current_t1: Node,
    current_t2: Node,
    deleted_taxa_per_iteration: List[Set[int]],
    solutions_dict_this_iter: Dict[Partition, List[Partition]],
    iteration_count: int,
) -> bool:
    """
    Identify jumping subtrees/taxa from solutions and remove them from trees.
    Attempts structural subtree removal first. If a solution does not map to a strict subtree, logic fails.

    Args:
        current_t1: Current tree 1 (modified in-place)
        current_t2: Current tree 2 (modified in-place)
        deleted_taxa_per_iteration: List to track removed taxa per iteration (modified in-place)
        solutions_dict_this_iter: Dictionary of solution partitions from this iteration
        iteration_count: Current iteration number

    Returns:
        should_break_loop: Boolean indicating if loop should break
    """
    # Collect all unique solution partitions
    unique_solutions: Set[Partition] = set()
    for solutions in solutions_dict_this_iter.values():
        unique_solutions.update(solutions)

    if not unique_solutions:
        return True  # Break loop

    # Sort solutions by size descending to encourage subtree removal over leaf picking
    unique_solutions_list = sorted(
        unique_solutions, key=lambda p: len(p.indices), reverse=True
    )

    # Track removed taxa PER TREE to handle asymmetry
    removed_indices_t1: Set[int] = set()
    removed_indices_t2: Set[int] = set()

    # For reporting (union of deleted)
    total_deleted_indices: Set[int] = set()
    count_removed_ops = 0

    # Process each solution
    for part in unique_solutions_list:
        all_indices = part.resolve_to_indices()
        total_deleted_indices.update(all_indices)

        # Apply to both trees independently
        for tree, removed_set in [
            (current_t1, removed_indices_t1),
            (current_t2, removed_indices_t2),
        ]:
            # Determine which taxa in this solution are still present
            target_indices = [i for i in all_indices if i not in removed_set]
            if not target_indices:
                continue  # All already removed

            # 1. Try structural removal (Subtree)
            # Note: find_node_by_split expects exact bitmask match.
            # If tree modified (some children removed), this will naturally return None.
            structural_node = tree.find_node_by_split(part)

            if structural_node:
                if structural_node.parent is None:
                    # User requested strict error handling
                    raise ValueError(
                        f"Solution {part} corresponds to the root node, which cannot be removed."
                    )

                tree.remove_subtree(structural_node, mode="stable")
                # If we removed the node corresponding to 'part',
                # we effectively removed all taxa in 'part'.
                removed_set.update(all_indices)
                count_removed_ops += 1
            else:
                raise ValueError(
                    f"Solution partition {part} does not correspond to a valid subtree/clade in the current tree structure "
                    f"and strict subtree deletion is enforced. (Tree encoding size: {len(tree.taxa_encoding)})"
                )

    if not total_deleted_indices:
        return True

    deleted_taxa_per_iteration.append(total_deleted_indices)

    if not jt_logger.disabled:
        jt_logger.debug(
            f"Iter {iteration_count}: Removed {count_removed_ops} structural units "
            f"({len(total_deleted_indices)} taxa)"
        )

    # Check if trees have enough leaves to continue
    t1_leaf_count = len(current_t1.get_leaves())
    t2_leaf_count = len(current_t2.get_leaves())
    if t1_leaf_count < 2 or t2_leaf_count < 2:
        return True  # Break loop

    return False  # Continue loop
