from typing import List, Set, Tuple, Dict
import functools
from brancharchitect.tree import Node
from brancharchitect.leaforder.circular_distances import (
    circular_distance,
    circular_distance_for_node_subset,
    circular_distance_based_on_reference,
    circular_distance_tree_pair,
)

rotation_split_history = dict()

##################################################
#           Classification Logic
##################################################


def classify_subtrees_using_set_ops(
    tree_ref: Node, tree_target: Node
) -> Dict[Node, str]:
    """
    For each internal node in tree_target, classify as:
      - 'full-common'   : node's split is in tree_ref & entire subtree is in tree_ref
      - 'common-changed': node's split is in tree_ref but subtree partially differs
      - 'full-unique'   : node's split not in tree_ref, subtree fully outside
      - 'unique-changed': partial overlap
    """
    Sref = tree_ref.to_splits()
    Stgt = tree_target.to_splits()
    S_common = Sref & Stgt

    subtree_splits_map: Dict[Node, Set[Tuple[int, ...]]] = {}

    def collect_subtree_splits(node: Node) -> Set[Tuple[int, ...]]:
        subs = set()
        for ch in node.children:
            subs |= collect_subtree_splits(ch)
        if node.children:
            subs.add(node.split_indices)
        subtree_splits_map[node] = subs
        return subs

    collect_subtree_splits(tree_target)

    classification_map: Dict[Node, str] = {}
    for node, splits in subtree_splits_map.items():
        node_in_ref = node.split_indices in Sref
        fully_in_ref = splits.issubset(S_common)
        fully_out_ref = splits.isdisjoint(S_common)

        if node_in_ref and fully_in_ref:
            classification_map[node] = "full-common"
        elif node_in_ref and not fully_in_ref:
            classification_map[node] = "common-changed"
        elif (not node_in_ref) and fully_out_ref:
            classification_map[node] = "full-unique"
        else:
            classification_map[node] = "unique-changed"

    return classification_map


##################################################
#        Splits Info + S-Edge
##################################################


def get_splits_info(tree1: Node, tree2: Node):
    """
    Return (unique2, common, s_edges_sorted).
    'unique2' are splits in tree2 not in tree1,
    's_edges' are those common splits that have at least one child unique to tree2.
    """
    s1 = tree1.to_splits()
    s2 = tree2.to_splits()
    common = s1 & s2
    unique2 = s2 - s1

    s_edges = []
    for sp in common:
        node = tree2.find_node_by_split(sp)
        if node and node.children:
            if any(
                ch.split_indices in unique2 for ch in node.children if ch.split_indices
            ):
                s_edges.append(sp)

    s_edges_sorted = sorted(s_edges, key=lambda x: len(x))
    sorted_unique2 = sorted(unique2, key=lambda x: len(x))
    return sorted_unique2, common, s_edges_sorted


##################################################
#          Local Rotation Tools
##################################################


def try_node_reversal_global(
    node: Node,
    tree: Node,
    initial_dist: float,
    reference_order: Tuple[str, ...],
    rotated_splits: Set[Tuple[int, ...]],
):
    """
    Swap node's children, measure full-tree distance vs. reference_order.
    Revert if no improvement; otherwise record node.split in rotated_splits.
    """
    node.swap_children()
    new_dist = circular_distance_based_on_reference(tree, reference_order)
    if new_dist < initial_dist:
        rotated_splits.add(node.split_indices)
        node.invalidate_current_order_cache()
        return True, new_dist
    else:
        node.swap_children()
        return False, initial_dist


def try_node_reversal_local(
    node: Node,
    tree: Node,
    initial_dist: float,
    reference_order: Tuple[str, ...],
    rotated_splits: Set[Tuple[int, ...]],
):
    """
    Swap node's children, measure local (subtree) distance. Revert if no improvement.
    """
    node.swap_children()
    new_dist = circular_distance_for_node_subset(tree, reference_order, node)
    if new_dist < initial_dist:
        rotated_splits.add(node.split_indices)
        node.invalidate_current_order_cache()
        return True, new_dist
    else:
        node.swap_children()
        return False, initial_dist


##################################################
#  Unique & S-Edge Rotation
##################################################


def optimize_unique_splits(
    tree1: Node,
    tree2: Node,
    reference_order: Tuple[str, ...],
    rotated_splits: Set[Tuple[int, ...]],
) -> bool:
    """
    For each 'unique' split in tree2, attempt a local reversal.
    """
    unique2, _, _ = get_splits_info(tree1, tree2)
    any_improvement = False
    for sp in unique2:
        node = tree2.find_node_by_split(sp)
        if node and node.children:
            init_dist = circular_distance_for_node_subset(tree2, reference_order, node)
            improved, _ = try_node_reversal_local(
                node, tree2, init_dist, reference_order, rotated_splits
            )
            # improved, _ = try_node_reversal_global(
            #     node, tree2, init_dist, reference_order, rotated_splits
            # )
            if improved:
                any_improvement = True
    return any_improvement


def optimize_s_edge_splits(tree1: Node, tree2: Node, reference_order, rotated_splits):
    _, _, s_edges_sorted = get_splits_info(tree1, tree2)
    current_dist = circular_distance_based_on_reference(tree2, reference_order)
    any_improvement = False

    for sp in s_edges_sorted:
        node = tree2.find_node_by_split(sp)
        if node and node.children:
            # Save original child order
            original_children = node.children[:]

            # -- Test global flip --
            node.swap_children()
            dist_g = circular_distance_based_on_reference(tree2, reference_order)
            # revert
            node.children = original_children[:]

            # -- Test local flip --
            node.swap_children()
            dist_l = circular_distance_based_on_reference(tree2, reference_order)
            # revert
            node.children = original_children[:]

            # Compare dist_g vs dist_l vs current_dist
            best_flip_dist = min(dist_g, dist_l, current_dist)
            if best_flip_dist < current_dist:
                # Decide which flip was best
                if dist_g < dist_l:
                    # do global flip for real
                    node.swap_children()
                    node.invalidate_current_order_cache()
                    new_dist = dist_g
                else:
                    # do local flip for real
                    node.swap_children()
                    node.invalidate_current_order_cache()
                    new_dist = dist_l

                current_dist = new_dist
                any_improvement = True
                rotated_splits.add(sp)
            else:
                # keep the node's orientation as original
                node.children = original_children[:]

    return any_improvement


def optimize_common_splits(
    tree1: Node,
    tree2: Node,
    reference_order: Tuple[str, ...],
    rotated_splits: Set[Tuple[int, ...]],
) -> bool:
    """
    Attempt to reorder node1's children to match node2 for any s-edge splits,
    if it leads to better distance for that pair.
    """
    if reference_order:
        tree1.reorder_taxa(reference_order)
        tree2.reorder_taxa(reference_order)

    unique2, common_splits, _ = get_splits_info(tree1, tree2)
    initial_dist = circular_distance_tree_pair(tree1, tree2)
    any_improvement = False
    curr_distance = initial_dist

    # We'll consider s-edge splits in 'common_splits'
    s_edges = []
    for sp in common_splits:
        node_in_tree2 = tree2.find_node_by_split(sp)
        if node_in_tree2 and node_in_tree2.children:
            if any(
                ch.split_indices in unique2
                for ch in node_in_tree2.children
                if ch.split_indices
            ):
                s_edges.append(sp)

    s_edges_sorted = sorted(s_edges, key=lambda x: len(x))
    for sp in s_edges_sorted:
        node1_s = tree1.find_node_by_split(sp)
        node2_s = tree2.find_node_by_split(sp)
        if not (node1_s and node2_s and node1_s.children and node2_s.children):
            continue

        improved, new_dist = try_reorder_node_children_to_match(
            node1_s, node2_s, curr_distance
        )
        if improved and new_dist < curr_distance:
            curr_distance = new_dist
            rotated_splits.add(sp)
            any_improvement = True

    return any_improvement


##################################################
#   Reorder Node Children to Match
##################################################


def try_reorder_node_children_to_match(
    node1: Node, node2: Node, current_distance: float
) -> Tuple[bool, float]:
    """
    Attempt reordering node1's children to match node2's child leaf-subsets.
    """
    ref_sets = [frozenset(ch.get_current_order()) for ch in node2.children]
    node1_map = {frozenset(ch.get_current_order()): ch for ch in node1.children}
    original_children = node1.children[:]
    reordered = []

    for sset in ref_sets:
        match = node1_map.get(sset)
        if match:
            reordered.append(match)
    unmatched = [c for c in original_children if c not in reordered]
    reordered.extend(unmatched)

    if reordered == original_children:
        return False, current_distance

    node1.children = reordered
    all_leaves = node1.get_current_order()
    if len(all_leaves) != len(set(all_leaves)):
        # revert if duplication found
        node1.children = original_children
        return False, current_distance

    new_dist = circular_distance_tree_pair(node1, node2)
    if new_dist < current_distance:
        return True, new_dist
    else:
        node1.children = original_children
        return False, current_distance


##################################################
#  Orientation Map + reorder_tree_if_full_common
##################################################


def reorder_tree_if_full_common(
    reference_tree: Node,
    target_tree: Node,
    orientation_map: Dict[Tuple[int, ...], List[frozenset]],
) -> Dict[Tuple[int, ...], List[frozenset]]:
    """
    For each split in orientation_map, if target_tree sees it as 'full-common',
    reorder children to match reference_tree. If not, remove that split from orientation_map.
    We keep optimizing other splits.
    """
    if not orientation_map:
        return orientation_map

    classification_map = classify_subtrees_using_set_ops(reference_tree, target_tree)
    splits_to_remove = []

    for sp, child_leafsets in orientation_map.items():
        node_in_tgt = target_tree.find_node_by_split(sp)
        node_in_rf = reference_tree.find_node_by_split(sp)
        node_class = classification_map.get(node_in_tgt, None)
        if node_class == "full-common":
            node_in_tgt.reorder_taxa(node_in_rf.get_current_order())
        else:
            # not full-common => remove
            splits_to_remove.append(sp)
    for sp in splits_to_remove:
        orientation_map.pop(sp, None)
    return orientation_map


##################################################
#   Forward & Backward Orientation Prop
##################################################
def build_orientation_map(
    tree: Node, rotated_splits: Set[Tuple[int, ...]]
) -> Dict[Tuple[int, ...], List[frozenset]]:
    """
    For each rotated split in `tree`, gather final child orientation
    as a list of leaf-subsets in order.
    """
    orientation_map = {}
    for sp in rotated_splits:
        node = tree.find_node_by_split(sp)
        orientation_map[sp] = [
            frozenset(ch.get_current_order()) for ch in node.children
        ]
    return orientation_map


def propagate_orientation_forward(
    trees: List[Node], i: int, rotated_splits: Set[Tuple[int, ...]]
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
    trees: List[Node], i: int, rotated_splits: Set[Tuple[int, ...]]
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
    reference_tree: Node, target_tree: Node, backward: bool = False
) -> Tuple[bool, Set[Tuple[int, ...]]]:
    """
    Perform local optimizations in tree2 for unique & s-edge splits,
    return (improved_any, set_of_splits_rotated_in_tree2).
    """
    rotated_splits_in_target = set()
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


def update_rotation_split_history(i, j, rotated_splits, improved):
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
    improved_any = False
    n = len(trees)

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


def improve_single_pair_classic(
    tree1: Node, tree2: Node, rotation_functions: List
) -> Tuple[bool, Node, Node]:
    """
    Applies each rotation function in sequence to (tree1, tree2).
    We pass a dummy 'rotated_splits' because the classic approach
    does not track orientation across pairs.
    """

    ref_order = tree1.get_current_order()
    best_dist = circular_distance(ref_order, tree2.get_current_order())
    improved = False

    dummy_rotated_splits: Set[Tuple[int, ...]] = set()

    for func in rotation_functions:
        func(tree1, tree2, ref_order, dummy_rotated_splits)
        curr_dist = circular_distance(ref_order, tree2.get_current_order())
        if curr_dist < best_dist:
            best_dist = curr_dist
            improved = True

    return improved


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


def perform_one_iteration_classic(
    trees: List[Node], rotation_functions: List, optimize_two_side: bool = False
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
        improved = improve_single_pair_classic(
            trees[i], trees[i + 1], rotation_functions
        )

        if improved:
            overall_improvement = True

    return overall_improvement


def smooth_order_of_trees_classic(
    trees: List[Node],
    rotation_functions: List,
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

            backward_improved = perform_one_iteration_classic(
                trees, rotation_functions, optimize_two_side
            )
            trees.reverse()

        if not (forward_improved or backward_improved):
            break


# #############################
# EXAMPLE USAGE
##################################################
"""
# 1) The classic approach:
# Define local rotation func
rotation_funcs = [
    optimize_unique_splits,
    optimize_s_edge_splits,
    optimize_common_splits,
]
smooth_order_of_trees_classic(
    trees,
    rotation_funcs,
    n_iterations=5,
    optimize_two_side=False,
    backward=True
)

# 2) The classification-based approach with forward+backward propagation:
smooth_order_unique_sedge_both(trees, n_iterations=5)
"""
