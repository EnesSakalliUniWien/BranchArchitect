from typing import Dict, List, Set

from brancharchitect.elements.partition import Partition
from brancharchitect.tree import Node


def verify_mapped_solutions_prune(
    original_tree1: Node,
    original_tree2: Node,
    current_t1: Node,
    current_t2: Node,
    mapped_solutions_dict: Dict[Partition, List[Partition]],
) -> None:
    """
    Validate mapped solutions by pruning deep copies of the original trees and
    asserting they match the current pruned trees.
    """
    t1_copy = original_tree1.deep_copy()
    t2_copy = original_tree2.deep_copy()

    apply_solution_partitions(t1_copy, t2_copy, mapped_solutions_dict)

    if t1_copy != current_t1 or current_t1 != current_t2 and t2_copy != current_t2:
        t1_expected = current_t1.to_splits()
        t1_actual = t1_copy.to_splits()
        t2_expected = current_t2.to_splits()
        t2_actual = t2_copy.to_splits()
        t1_missing = {tuple(s.indices) for s in (t1_expected - t1_actual)}
        t1_extra = {tuple(s.indices) for s in (t1_actual - t1_expected)}
        t2_missing = {tuple(s.indices) for s in (t2_expected - t2_actual)}
        t2_extra = {tuple(s.indices) for s in (t2_actual - t2_expected)}
        raise ValueError(
            "Mapped-solution verification failed: pruned originals do not match "
            "current trees. "
            f"t1_missing={t1_missing} t1_extra={t1_extra} "
            f"t2_missing={t2_missing} t2_extra={t2_extra}"
        )


def apply_solution_partitions(
    tree1: Node,
    tree2: Node,
    solutions_dict: Dict[Partition, List[Partition]],
) -> None:
    unique_solutions: Set[Partition] = set()
    for solutions in solutions_dict.values():
        unique_solutions.update(solutions)

    if not unique_solutions:
        return

    unique_solutions_list = sorted(
        unique_solutions, key=lambda p: len(p.indices), reverse=True
    )

    removed_indices_t1: Set[int] = set()
    removed_indices_t2: Set[int] = set()

    for part in unique_solutions_list:
        all_indices = part.resolve_to_indices()

        for tree, removed_set in [
            (tree1, removed_indices_t1),
            (tree2, removed_indices_t2),
        ]:
            target_indices = [i for i in all_indices if i not in removed_set]
            if not target_indices:
                continue

            structural_node = tree.find_node_by_split(part)
            if structural_node is None:
                raise ValueError(
                    f"Mapped solution partition {part} does not correspond to a valid subtree "
                    "in verification tree."
                )
            if structural_node.parent is None:
                raise ValueError(
                    f"Mapped solution partition {part} corresponds to the root, "
                    "which cannot be removed."
                )

            tree.remove_subtree(structural_node, mode="stable")
            removed_set.update(all_indices)
