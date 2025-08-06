from typing import Sequence
from typing import List, Tuple, Callable
from brancharchitect.tree import Node
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.elements.partition import Partition
from brancharchitect.leaforder.circular_distances import (
    circular_distance,
)
from brancharchitect.leaforder.rotation_functions import (
    optimize_unique_splits,
    optimize_s_edge_splits,
)


def improve_single_pair_classic(
    tree1: Node,
    tree2: Node,
    rotation_functions: Sequence[
        Callable[[Node, Node, Tuple[str, ...], PartitionSet[Partition]], bool]
    ] = (
        optimize_s_edge_splits,
        optimize_unique_splits,
    ),
) -> bool:
    """
    Applies each rotation function in sequence to (tree1, tree2).
    We pass a dummy 'rotated_splits' because the classic approach
    does not track orientation across pairs.
    """

    ref_order: Tuple[str, ...] = tree1.get_current_order()
    best_dist: float = circular_distance(ref_order, tree2.get_current_order())
    improved: bool = False

    for func in rotation_functions:
        func(tree1, tree2, ref_order)
        curr_dist = circular_distance(ref_order, tree2.get_current_order())
        if curr_dist < best_dist:
            best_dist = curr_dist
            improved = True

    return improved


def smooth_order_of_trees_classic(
    trees: List[Node],
    rotation_functions: Sequence[
        Callable[[Node, Node, Tuple[str, ...], PartitionSet[Partition]], bool]
    ] = (
        optimize_s_edge_splits,
        optimize_unique_splits,
    ),
    n_iterations: int = 3,
    optimize_two_side: bool = False,
    backward: bool = False,
):
    """
    The local (classic) approach with up to n_iterations,
    optionally enabling 'optimize_two_side' for each pair,
    and a backward pass that reverses the entire list.
    We stop if no improvement in a full iteration.
    """
    for _ in range(n_iterations):
        forward_improved = perform_one_iteration_classic(
            trees, rotation_functions, optimize_two_side
        )

        backward_improved = False

        if backward:
            trees.reverse()

            backward_improved: bool = perform_one_iteration_classic(
                trees, rotation_functions, optimize_two_side
            )

            trees.reverse()

        if not (forward_improved or backward_improved):
            break


def perform_one_iteration_classic(
    trees: List[Node],
    rotation_functions: Sequence[
        Callable[[Node, Node, Tuple[str, ...], PartitionSet[Partition]], bool]
    ] = (
        optimize_s_edge_splits,
        optimize_unique_splits,
    ),
    optimize_two_side: bool = False,
) -> bool:
    """
    One iteration of the classic approach over all adjacent pairs,
    optionally doing two-sided optimization for each pair.
    """
    overall_improvement = False
    n = len(trees)
    for i in range(n - 1):
        # optional: optimize from T[i+1] to T[i]
        if optimize_two_side:
            improved = improve_single_pair_classic(
                trees[i + 1], trees[i], rotation_functions
            )
            if improved:
                overall_improvement = True

        # normal direction: T[i] to T[i+1]
        improved: bool = improve_single_pair_classic(
            trees[i], trees[i + 1], rotation_functions
        )

        if improved:
            overall_improvement = True

    return overall_improvement
