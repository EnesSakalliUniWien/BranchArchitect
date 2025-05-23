from typing import List, Tuple, Dict
from brancharchitect.tree import Node
from brancharchitect.partition_set import PartitionSet, Partition
from brancharchitect.leaforder.rotation_functions import (
    optimize_unique_splits,
    optimize_s_edge_splits,
)
from brancharchitect.leaforder.tree_order_utils import (
    build_orientation_map,
    reorder_tree_if_full_common,
)
rotation_split_history = dict()



def propagate_orientation_forward(
    trees: List[Node], i: int, rotated_splits: PartitionSet
):
    """
    T[i+1] => T[i+2], T[i+3], ...
    Reorder if full-common, or drop that split from the orientation map.
    """
    if i < 0 or i >= len(trees) - 1:
        return
    ref_tree = trees[i + 1]
    orientation_map = build_orientation_map(ref_tree, rotated_splits)
    for j in range(i + 2, len(trees)):
        orientation_map = reorder_tree_if_full_common(
            ref_tree, trees[j], orientation_map
        )
        if not orientation_map:
            break


def propagate_orientation_backward(
    trees: List[Node], i: int, rotated_splits: PartitionSet
):
    """
    T[i] => T[i-1], T[i-2], ...
    Reorder if full-common, or drop that split from the orientation map.
    """
    if i <= 0 or i >= len(trees):
        return
    ref_tree = trees[i]
    orientation_map = build_orientation_map(ref_tree, rotated_splits)

    for j in range(i - 1, -1, -1):
        orientation_map = reorder_tree_if_full_common(
            ref_tree, trees[j], orientation_map
        )
        if not orientation_map:
            break


##################################################
#   Classification Approach: Local Flips + Prop
##################################################


def improve_unique_sedge_pair(
    reference_tree: Node, target_tree: Node
) -> Tuple[bool, PartitionSet]:
    """
    Perform local optimizations in tree2 for unique & s-edge splits,
    return (improved_any, set_of_splits_rotated_in_tree2).
    """
    rotated_splits_in_target: PartitionSet = PartitionSet()
    ref_order = reference_tree.get_current_order()

    # optimize unique splits
    improved_unique = optimize_unique_splits(
        reference_tree, target_tree, ref_order, rotated_splits_in_target
    )
    # optimize s-edge splits
    improved_sedge = optimize_s_edge_splits(
        reference_tree, target_tree, ref_order, rotated_splits_in_target
    )

    return ((improved_unique or improved_sedge), rotated_splits_in_target)


def update_rotation_split_history(i: int, j: int, rotated_splits, improved: bool):
    if (i, j) not in rotation_split_history:
        rotation_split_history[(i, j)] = {
            "rotated_splits": rotated_splits,
            "improved": improved,
        }
    else:
        rotation_split_history[(i, j)]["improved"] = {
            "improved": improved,
        }


def forward_pass_unique_sedge_prop(trees: List[Node]) -> bool:
    improved_any: bool = False
    n: int = len(trees)

    for i in range(n - 1):
        improved, rotated_splits = improve_unique_sedge_pair(trees[i], trees[i + 1])
        rotation_split_history[(i, i + 1)] = {
            "rotated_splits": rotated_splits,
            "improved": improved,
        }
        # (Option A) If you want immediate propagation, uncomment:
        if improved:
            improved_any = True
            propagate_orientation_forward(trees, i, rotated_splits)

    # (Option B) Or do the propagation afterward, but still track improvements:
    # for (i, j), history in rotated_splits_history.items():
    #     # print(f"Tree pair: {i} -> {j}, improved: {history['improved']}")
    #     # print(f"Rotated splits: {history['rotated_splits']}")
    #     if history["improved"]:
    #         improved_any = True  # <-- Mark improvement!
    #         propagate_orientation_forward(trees, i, history["rotated_splits"])

    return improved_any


def backward_pass_unique_sedge_prop(trees: List[Node]) -> bool:
    """
    Backward pass from T[n-1]..T[0].
    Each time a local rotation occurs in T[i], we propagate orientation backward.
    """
    improved_any = False
    n = len(trees)
    for i in range(n - 1, 0, -1):
        improved, rotated_splits = improve_unique_sedge_pair(trees[i], trees[i - 1])
        if improved:
            improved_any = True
            propagate_orientation_backward(trees, i, rotated_splits)
    return improved_any


def smooth_order_unique_sedge_both(trees: List[Node], n_iterations: int = 3):
    """
    The classification-based approach with both forward & backward passes each iteration.
    If a split is no longer 'full-common' in some tree, we remove it from the orientation map
    but keep going for the rest.
    Stop if no improvement after a forward+back pass.
    """
    for _ in range(n_iterations):
        forward_pass_unique_sedge_prop(trees)
        backward_pass_unique_sedge_prop(trees)

        trees.reverse()
        forward_pass_unique_sedge_prop(trees)
        backward_pass_unique_sedge_prop(trees)
        trees.reverse()


##################################################
#   Classic Approach (Local)
##################################################


def smooth_order_unique_sedge(
    trees: List[Node], n_iterations: int = 3, backward: bool = False
):
    """
    Repeatedly do forward passes, if backward=True also do reverse passes,
    using the classification-based orientation approach + early stopping
    for splits that fail 'full-common' in any future tree.
    """
    for _ in range(n_iterations):
        forward_impr = forward_pass_unique_sedge_prop(trees)
        # forward_impr = backward_pass_unique_sedge_prop(trees)

        back_impr = False
        if backward:
            trees.reverse()
            back_impr = forward_pass_unique_sedge_prop(trees)
            # back_impr = backward_pass_unique_sedge_prop(trees)
            trees.reverse()
        if not (forward_impr or back_impr):
            break