from typing import List, Optional, Set, Tuple
from brancharchitect.tree import Node, SplitIndices
from brancharchitect.tree_order_optimisation_global import find_node_by_split
from brancharchitect.leaf_order_distances import (
    circular_distance_tree_pair,
    circular_distance_based_on_reference,
)

#####################################################
#          reverse_unique_splits_by_check
#####################################################


def attempt_node_reversal(
    current_node: "Node",
    tree1: "Node",
    tree2: "Node",
    initial_distance: float,
    common_splits: Set["SplitIndices"],
    reference_order: List,
) -> Tuple[bool, float]:
    """
    Attempt to reverse the children of a single node if it's not in common_splits.
    Check if it improves the distance between tree2 and tree1. If not improved, revert.

    Args:
        current_node (Node): The node in tree2 we consider reversing.
        tree1 (Node): Reference tree for distance measurement.
        tree2 (Node): The tree we're trying to improve.
        initial_distance (float): Current best distance before attempting reversal.
        common_splits (Set[SplitIndices]): Splits considered common; we only reverse if node not in these.

    Returns:
        (improved: bool, new_initial_distance: float)
        improved indicates if the reversal helped.
        new_initial_distance updates if improvement found.

    Example:
        >>> improved, new_dist = attempt_node_reversal(current_node, tree1, tree2, initial_dist, common_splits)
    """

    # Check if current_node has children and is considered unique (not in common_splits)
    if current_node.children and current_node.split_indices not in common_splits:
        # Backup original children order
        original_children = current_node.children[:]

        # Reverse children
        current_node.children.reverse()

        # Check new distance after reversal
        new_distance = reference_order(tree1, tree2)

        if new_distance < initial_distance:
            # Improvement found
            return True, new_distance
        else:
            # No improvement, revert to original order
            current_node.children = original_children
            return False, initial_distance
    else:
        # If no children or it's a common split, do nothing
        return False, initial_distance


def reverse_unique_splits_by_check(
    node: "Node",
    tree1: "Node",
    tree2: "Node",
    reference_order: List,
    common_splits: Optional[Set["SplitIndices"]] = None,
) -> bool:
    """
    Attempt incremental improvements by reversing unique splits in `tree2` to reduce circular distance to `tree1`.
    Uses a stack-based DFS to examine each descendant node.
    If a node is unique (not in common_splits), try reversing its children and keep changes only if it helps.

    Args:
        node (Node): Subtree root in tree2 to process.
        tree1 (Node): Reference tree.
        tree2 (Node): The tree to improve by reversing unique splits.
        common_splits (Set[SplitIndices], optional): Common splits; nodes in these won't be reversed.

    Returns:
        bool: True if an improvement was made, otherwise False.

    Example:
        >>> improved = reverse_unique_splits_by_check(node, tree1, tree2, common_splits)
        # If improved, tree2 is modified in place to reflect better ordering.
    """

    if common_splits is None:
        common_splits = set()

    # Measure initial distance before any reversal attempts
    initial_distance = circular_distance_based_on_reference(reference_order, tree2)

    # any_improvement tracks if we made improvements at any node
    any_improvement = False

    # We'll do a stack-based DFS over subtree
    stack = [node]

    while stack:
        current_node = stack.pop()
        current_node.swap_children()
        # Attempt reversal at this node
        improved, initial_distance = attempt_node_reversal(
            current_node, tree1, tree2, initial_distance, common_splits
        )
        if improved:
            any_improvement = True
            current_node.swap_children()

        # If node has children, push them onto stack to check deeper
        if current_node.children:
            stack.extend(current_node.children)

    return any_improvement


#####################################################
#              rotate_unique_splits
#####################################################


def collect_splits_info(tree1: "Node", tree2: "Node", with_common_splits: bool):
    """
    Collect information about unique and common splits for rotate_unique_splits.

    Args:
        tree1 (Node): Reference tree.
        tree2 (Node): Target tree.

    Returns:
        (unique_splits2: Set[SplitIndices], common_splits: Set[SplitIndices])

    Example:
        >>> unique_splits2, common_splits = collect_splits_info_for_unique_rotation(tree1, tree2, True)
    """
    splits1 = tree1.to_splits()
    splits2 = tree2.to_splits()

    unique_splits2 = splits2 - splits1
    common_splits = (splits2 & splits1) if with_common_splits else set()
    return unique_splits2, common_splits


def attempt_reversals_on_unique_splits(
    tree1: "Node",
    tree2_copy: "Node",
    unique_splits2: Set["SplitIndices"],
    common_splits: Set["SplitIndices"],
    reference_order: List,
) -> bool:
    """
    Attempt reversing unique splits on each unique split node in tree2_copy using reverse_unique_splits_by_check.

    Args:
        tree1 (Node): Reference tree.
        tree2_copy (Node): Working copy of tree2.
        unique_splits2 (Set[SplitIndices]): Unique splits in tree2.
        common_splits (Set[SplitIndices]): Common splits.

    Returns:
        bool: True if any improvement was made at any unique split node.
    """
    any_improvement = False
    for split in unique_splits2:
        # Find the corresponding node in tree2_copy for this split
        node = find_node_by_split(tree2_copy, split)
        if node and node.children:
            # Try reversing unique splits in this node's subtree
            improved = reverse_unique_splits_by_check(
                node, tree1, tree2_copy, common_splits, reference_order=reference_order
            )
            if improved:
                any_improvement = True
    return any_improvement


def rotate_unique_splits(
    tree1: "Node", reference_order: List, tree2: "Node", with_common_splits: bool = True
) -> bool:
    """
    Rotate unique splits in `tree2` to improve alignment with `tree1`.
    Uses `reverse_unique_splits_by_check` to attempt incremental improvements.

    Args:
        tree1 (Node): Reference tree.
        tree2 (Node): Target tree to potentially improve.
        with_common_splits (bool): If True, consider common splits. If False, treat all splits in tree2 as unique.

    Returns:
        bool: True if any improvement was made, otherwise False.

    Example:
        >>> improved = rotate_unique_splits(tree1, tree2, with_common_splits=True)
        # If improved, tree2 is updated in place.
    """
    # Collect information on unique and common splits
    unique_splits2, common_splits = collect_splits_info(
        tree1, tree2, with_common_splits
    )

    # Measure initial distance
    initial_distance = circular_distance_based_on_reference(reference_order, tree2)
    # Work on a copy so we only commit if improvements happen
    tree2_copy = tree2.deep_copy()

    # Attempt reversals on nodes with unique splits
    any_improvement = attempt_reversals_on_unique_splits(
        tree1, tree2, unique_splits2, common_splits, reference_order=reference_order
    )

    # Check if overall improvement is achieved
    new_distance = circular_distance_based_on_reference(
        reference_order=reference_order, target_tree=tree2
    )

    if any_improvement and new_distance < initial_distance:
        # Commit changes to tree2
        tree2.__dict__.update(tree2_copy.__dict__)
        return True
    return False


#####################################################
#               check_if_s_edge
#####################################################


def check_if_s_edge(
    node: "Node",
    unique_splits2: Set["SplitIndices"],
    common_splits: Set["SplitIndices"],
) -> bool:
    """
    Check if `node` corresponds to a common split that includes at least one unique split child (an 's-edge').

    Args:
        node (Node): The node to check.
        unique_splits2 (Set[SplitIndices]): Unique splits in second tree.
        common_splits (Set[SplitIndices]): Common splits shared by both trees.

    Returns:
        bool: True if node is an s-edge, otherwise False.

    Example:
        >>> is_edge = check_if_s_edge(node, unique_splits2, common_splits)
    """

    # Node must be a common split to be considered
    if node.split_indices not in common_splits:
        return False

    # Check if at least one child is a unique split
    return any(
        child.split_indices in unique_splits2
        for child in node.children
        if child.split_indices
    )


#####################################################
#                rotate_s_edge
#####################################################


def attempt_s_edge_rotations_at_node(
    node: "Node",
    tree1: "Node",
    tree2_copy: "Node",
    initial_distance: float,
) -> float:
    """
    Attempt to rotate children at this node if it's an s-edge. If no improvement, revert changes.

    Args:
        node (Node): The node where we attempt rotation.
        tree1 (Node): Reference tree.
        tree2_copy (Node): Copy of target tree.
        unique_splits2 (Set[SplitIndices]): Unique splits in tree2.
        common_splits (Set[SplitIndices]): Common splits.

    Returns:
        float: Updated best distance after attempting rotation at this node.
    """
    # Backup original children
    # Reverse children to try improvement
    node.swap_children()
    new_distance = circular_distance_tree_pair(tree1, tree2_copy)

    if new_distance < initial_distance:
        # Improvement found, keep the rotation
        return new_distance
    else:
        # No improvement, revert to original children order
        node.swap_children()
        return initial_distance


def rotate_s_edge(tree1: "Node", tree2: "Node", reference_order: List) -> bool:
    """
    Rotate children at s-edge nodes in `tree2` to improve alignment with `tree1`.
    Reverts changes if no improvement is found.

    Args:
        tree1 (Node): Reference tree.
        tree2 (Node): Target tree.

    Returns:
        bool: True if any improvement was made, otherwise False.

    Example:
        >>> improved = rotate_s_edge(tree1, tree2)
    """

    # Determine common and unique splits
    unique_splits2, common_splits = collect_splits_info(
        tree1, tree2, with_common_splits=True
    )

    # Measure initial distance and work on a copy
    initial_distance = circular_distance_tree_pair(tree1, tree2)
    tree2_copy = tree2.deep_copy()

    # Sort common_splits by their length (or complexity)
    for common_split in sorted(common_splits, key=lambda x: len(x)):
        # Find corresponding node in tree2_copy
        node = find_node_by_split(tree2_copy, common_split)
        if check_if_s_edge(node, unique_splits2, common_splits):

            # Attempt s-edge rotation at this node
            initial_distance = attempt_s_edge_rotations_at_node(
                node, tree1, tree2_copy, reference_order, initial_distance
            )

    # After trying all s-edge rotations, check if improved
    final_distance = circular_distance_tree_pair(tree1, tree2)
    if final_distance < circular_distance_tree_pair(tree1, tree2):
        #     # Commit improvements
        #     tree2.__dict__.update(tree2_copy.__dict__)
        return True
    return False


#####################################################
#       rotate_common_splits_with_match
#####################################################


def reorder_node_children_to_match(
    node1: "Node", node2: "Node", initial_distance: float, tree1: "Node", tree2: "Node"
) -> Tuple[bool, float]:
    """
    Attempt to reorder node1's children to match node2's children order.
    Keep changes if it reduces the distance.

    Args:
        node1 (Node): Node in tree1 to reorder.
        node2 (Node): Corresponding node in tree2 to match.
        initial_distance (float): Current best distance
        tree1 (Node): Reference tree being modified
        tree2 (Node): Tree to match order from

    Returns:
        (improved: bool, new_initial_distance: float)
    """
    # Map children of node1 by their leaf sets
    split_to_child_node1 = {
        frozenset(ch.get_current_order()): ch for ch in node1.children
    }
    child_splits_node2 = [frozenset(ch.get_current_order()) for ch in node2.children]

    original_children = node1.children[:]
    new_children_order = []

    # Try to match node2's order
    for ds in child_splits_node2:
        if ds in split_to_child_node1:
            new_children_order.append(split_to_child_node1[ds])

    # Add any unmatched children at the end
    for ch in original_children:
        if ch not in new_children_order:
            new_children_order.append(ch)

    node1.children = new_children_order
    # Check leaf uniqueness
    leaf_names = node1.get_current_order()
    if len(leaf_names) != len(set(leaf_names)):
        # Duplicates introduced, revert
        node1.children = original_children
        return False, initial_distance

    # Check if distance improved
    new_distance = circular_distance_tree_pair(tree1, tree2)
    if new_distance < initial_distance:
        return True, new_distance
    else:
        # Revert if no improvement
        node1.children = original_children
        return False, initial_distance


def rotate_common_splits_with_match(tree1: "Node", tree2: "Node") -> bool:
    """
    Attempt to reorder children of common splits in tree1 to match tree2's order,
    keeping changes if they reduce the distance.

    Args:
        tree1 (Node): The tree to modify.
        tree2 (Node): The reference tree for desired order.

    Returns:
        bool: True if any improvement was made, otherwise False.

    Example:
        >>> improved = rotate_common_splits_with_match(tree1, tree2)
    """
    splits1 = tree1.to_splits()
    splits2 = tree2.to_splits()
    common_splits = splits1 & splits2

    initial_distance = circular_distance_tree_pair(tree1, tree2)
    any_improvement = False

    # For each common split, attempt to reorder node1's children to match node2
    for split in common_splits:
        node1 = find_node_by_split(tree1, split)
        node2 = find_node_by_split(tree2, split)
        if node1 and node2 and node1.children and node2.children:
            improved, initial_distance = reorder_node_children_to_match(
                node1, node2, initial_distance, tree1, tree2
            )
            if improved:
                any_improvement = True

    return any_improvement


#####################################################
#        ATTEMPT BREAK POINT ROTATION
#####################################################


def compute_topology_similarity(tree1: "Node", tree2: "Node") -> float:
    splits1 = tree1.to_splits()
    splits2 = tree2.to_splits()
    common = len(splits1 & splits2)
    total = len(splits1 | splits2)
    return common / total if total > 0 else 0


#####################################################
#           LOCAL PAIRWISE OPTIMIZATION
#####################################################


def improve_single_pair_dependent(
    tree1: "Node", tree2: "Node", rotation_functions: List
) -> Tuple[bool, "Node", "Node"]:
    """
    Try to improve a single pair (tree1, tree2) with given rotation functions.
    Each rotation builds upon previous improvements.

    Returns (improved, best_tree1, best_tree2).
    """
    current_t1 = tree1.deep_copy()
    current_t2 = tree2.deep_copy()
    current_distance = circular_distance_tree_pair(current_t1, current_t2)
    best_distance = current_distance
    best_pair = (current_t1, current_t2)
    improved = False

    # Try each rotation function
    for rotate_func in rotation_functions:
        # Try current direction
        temp_t1 = current_t1.deep_copy()
        temp_t2 = current_t2.deep_copy()
        rotate_func(temp_t1, temp_t2)
        dist = circular_distance_tree_pair(temp_t1, temp_t2)

        if dist < best_distance:
            best_distance = dist
            best_pair = (temp_t1, temp_t2)
            current_t1, current_t2 = temp_t1, temp_t2
            improved = True

        # Try reverse direction
        temp_t1 = current_t1.deep_copy()
        temp_t2 = current_t2.deep_copy()
        rotate_func(temp_t2, temp_t1)
        dist = circular_distance_tree_pair(temp_t1, temp_t2)

        if dist < best_distance:
            best_distance = dist
            best_pair = (temp_t1, temp_t2)
            current_t1, current_t2 = temp_t1, temp_t2
            improved = True

    return improved, best_pair[0], best_pair[1]


def improve_single_pair(
    tree1: "Node",
    tree2: "Node",
    rotation_functions: List,
    optimize_both_directions: bool = True,
) -> Tuple[bool, "Node", "Node"]:
    """
    Try to improve a single pair (tree1, tree2) with given rotation functions.
    If optimize_both_directions is False, only attempt (tree1 -> tree2).
    If True, also attempt (tree2 -> tree1).

    Args:
        tree1: First tree to optimize
        tree2: Second tree to optimize
        rotation_functions: List of rotation functions to apply
        optimize_both_directions (bool): Whether to attempt optimization in both directions.
    """
    current_t1 = tree1.deep_copy()
    current_t2 = tree2.deep_copy()
    current_distance = circular_distance_tree_pair(current_t1, current_t2)
    best_distance = current_distance
    best_pair = (current_t1, current_t2)
    improved = False

    reference_order_2 = current_t2.get_current_order()
    reference_order_1 = current_t1.get_current_order()

    # Attempt to optimize tree2 using best result from tree1 (reverse direction)
    if optimize_both_directions:
        for rotate_func in rotation_functions:
            temp_t1 = current_t1.deep_copy()
            temp_t2 = current_t2.deep_copy()
            # Attempt the reverse direction (tree2 -> tree1)
            rotate_func(temp_t2, temp_t1, reference_order_2)
            dist = circular_distance_based_on_reference(
                reference_order_2=reference_order_2, target_tree=temp_t1
            )
            if dist < best_distance:
                best_distance = dist
                best_pair = (temp_t1, temp_t2)
                current_t1, current_t2 = temp_t1, temp_t2
                improved = True

    reference_order_1 = current_t1.get_current_order()

    # Attempt to optimize tree1 (forward direction)
    for rotate_func in rotation_functions:
        temp_t1 = current_t1.deep_copy()
        temp_t2 = current_t2.deep_copy()

        # Attempt forward direction (tree1 -> tree2)
        rotate_func(temp_t1, temp_t2, reference_order_1)

        dist = circular_distance_based_on_reference(
            temp_t2, reference_order=reference_order_1
        )

        if dist < best_distance:
            best_distance = dist
            best_pair = (temp_t1, temp_t2)
            current_t1, current_t2 = temp_t1, temp_t2
            improved = True

    return improved, best_pair[0], best_pair[1]


def perform_one_iteration(
    trees: List["Node"], rotation_functions: List, optimize_both_directions: bool = True
) -> bool:
    """
    Perform one iteration of local improvements over all consecutive pairs.
    If optimize_both_directions is False, only try (tree1 -> tree2).
    Otherwise, try both (tree1 -> tree2) and (tree2 -> tree1).

    Args:
        trees: List of trees to optimize
        rotation_functions: List of rotation functions to apply
        optimize_both_directions (bool): Whether to attempt optimization in both directions.

    Returns:
        bool: True if any pair improved
    """
    num_trees = len(trees)
    overall_improvement = False

    for idx in range(num_trees - 1):
        tree1 = trees[idx]
        tree2 = trees[idx + 1]
        initial_distance = circular_distance_tree_pair(tree1, tree2)

        # Try first direction (tree1 -> tree2)
        improved1, t1a, t2a = improve_single_pair(
            tree1,
            tree2,
            rotation_functions,
            optimize_both_directions=optimize_both_directions,
        )
        dist1 = circular_distance_tree_pair(t1a, t2a) if improved1 else float("inf")

        # If we want both directions, try reverse direction (tree2 -> tree1)
        if optimize_both_directions:
            improved2, t2b, t1b = improve_single_pair(
                tree2,
                tree1,
                rotation_functions,
                optimize_both_directions=optimize_both_directions,
            )
            dist2 = circular_distance_tree_pair(t1b, t2b) if improved2 else float("inf")
        else:
            # No reverse direction attempt
            dist2 = float("inf")

        # Keep best improvement
        if dist1 < initial_distance or dist2 < initial_distance:
            if dist1 <= dist2:
                trees[idx] = t1a
                trees[idx + 1] = t2a
            else:
                trees[idx] = t1b
                trees[idx + 1] = t2b
            overall_improvement = True

    return overall_improvement


def local_pairwise_optimization(
    trees: List["Node"],
    max_iterations: int,
    rotation_functions: List,
    optimize_both_directions: bool = True,
):
    """
    Perform multiple iterations of local improvements on consecutive tree pairs.
    If optimize_both_directions is False, only optimize forward direction each iteration.
    """
    for iteration in range(max_iterations):
        overall_improvement = perform_one_iteration(
            trees, rotation_functions, optimize_both_directions
        )

        if not overall_improvement:
            print(
                f"No improvements in iteration {iteration + 1}, stopping initial optimization."
            )
            break


#####################################################
#            INITIAL REVERSAL & FINAL REFINEMENTS
#####################################################
def handle_initial_reversal(trees: List["Node"], optimize_from_behind: bool) -> bool:
    """
    If optimize_from_behind is True, reverse the tree list at the start.
    Return True if reversed, False otherwise.

    Args:
        trees: The tree list.
        optimize_from_behind: If True, reverse order now and restore later.

    Example:
        >>> reversed_order = handle_initial_reversal(trees, True)
    """
    if optimize_from_behind:
        trees.reverse()
        return True
    return False


def detect_topology_transitions(trees: List["Node"]) -> List[int]:
    """Detect indices where tree topology changes significantly."""
    transitions = []
    for i in range(len(trees) - 1):
        similarity = compute_topology_similarity(trees[i], trees[i + 1])
        if similarity < 0.8:  # Threshold for significant topology change
            transitions.append(i + 1)  # Transition occurs at i+1
    return transitions


def align_segment_to_tree(segment: List["Node"], target_tree: "Node"):
    """Align the segment of trees to have the same leaf ordering as the target_tree."""
    target_order = target_tree.get_current_order()
    for tree in segment:
        tree.reorder_taxa(target_order)


def attempt_transition_alignment(trees: List["Node"], idx: int):
    """Attempt to align the trees at the transition point to minimize distance."""
    if idx <= 0 or idx >= len(trees):
        return
    tree_prev = trees[idx - 1]
    tree_curr = trees[idx]

    rotation_functions = [
        rotate_common_splits_with_match,
        rotate_s_edge,
        rotate_unique_splits,
    ]

    # Try to improve the pair (tree_prev, tree_curr)
    improved, new_tree_prev, new_tree_curr = improve_single_pair(
        tree_prev, tree_curr, rotation_functions
    )

    if improved:
        trees[idx - 1] = new_tree_prev
        trees[idx] = new_tree_curr


def minimize_circular_distances(
    trees: List["Node"],
    max_iterations: int = 5,
    optimize_from_behind: bool = False,
    optimize_both_directions: bool = True,
):
    """
    Main function to minimize circular distances between consecutive trees.

    Args:
        trees (List[Node]): Sequence of trees.
        max_iterations (int): Max iterations for local improvements.
        optimize_from_behind (bool): If True, reverse order at start and revert at end.
        optimize_both_directions (bool): If True, attempt optimization in both directions.
    """
    # Handle optional reversal
    reversed_order = handle_initial_reversal(trees, optimize_from_behind)

    # Rotation strategies for local improvements
    rotation_functions = [
        rotate_s_edge,
        rotate_unique_splits,
        rotate_common_splits_with_match,
    ]

    # Local improvements with direction control
    local_pairwise_optimization(
        trees, max_iterations, rotation_functions, optimize_both_directions
    )

    # If we reversed initially, restore original order now
    if reversed_order:
        trees.reverse()


def minimize_circular_distances_with_transition_alignment(
    trees: List["Node"], optimize_both_directions: bool = True
):
    """
    Optimize trees by aligning segments around transition points to minimize distances.

    Args:
        trees (List[Node]): The list of trees to optimize.
        optimize_both_directions (bool): If True, attempts bidirectional optimization.
    """
    transitions = detect_topology_transitions(trees)
    n = len(trees)
    # Now pass the boolean to the minimize_circular_distances function
    minimize_circular_distances(
        trees, max_iterations=5, optimize_both_directions=optimize_both_directions
    )
    # Align segments before and after transition points
    start_idx = 0
    for idx in transitions:
        # Align the segment before the transition to the transition tree
        if start_idx < idx - 1:
            segment = trees[start_idx : idx - 1]
            target_tree = trees[idx - 1]
            align_segment_to_tree(segment, target_tree)
        start_idx = idx

    # Align the last segment after the last transition
    if start_idx < n:
        segment = trees[start_idx:]
        target_tree = trees[start_idx - 1] if start_idx > 0 else trees[start_idx]
        align_segment_to_tree(segment, target_tree)


def get_splits_for_trajectories(tree: "Node") -> Set["SplitIndices"]:
    """Get unique splits in a tree."""
    tree_pair_separated_splits = {}
    for i in range(len(tree) - 1):
        split_tree_i = tree[i].to_splits()
        split_tree_i1 = tree[i + 1].to_splits()
        unique_splits = split_tree_i1 - split_tree_i
        tree_pair_separated_splits[(i, i + 1)] = {
            "unique": unique_splits,
            "common": split_tree_i & split_tree_i1,
        }
