from brancharchitect.tree import Node
from typing import List
from .functional_tree import (
    FunctionalTree,
    ComponentSet,
    Component,
    build_functional_tree,
)
from brancharchitect.tree_interpolation.interpolation import (
    interpolate_tree,
)
from brancharchitect.jumping_taxa.algorithm_5.elemental import (
    argmax,
    count,
    argmin,
    size,
    filter_components_from_arms,
    map2,
    intersect,
    symm,
    cartesian,
    filter_,
    map1,
    remove_last_component_if_longer_than_one,
    union,
    reduce,
    cut,
    calculate_component_set,
    merge_sedges,
    remove_empty_lists,
    find_exact_max_intersection,
)


def get_ancestor_edge(t: FunctionalTree, c: ComponentSet) -> Node:
    return t._ancestor_edges[tuple(c)]


def algo5_partial_partial_cond(t1, t2):
    def cond(component):
        ancestor_edge1 = get_ancestor_edge(t1, component)
        ancestor_edge2 = get_ancestor_edge(t2, component)

        partial1 = is_partial_s_edge(t1, ancestor_edge1)
        partial2 = is_partial_s_edge(t2, ancestor_edge2)

        anti1 = is_anti_s_edge(t1, ancestor_edge1)
        anti2 = is_anti_s_edge(t2, ancestor_edge2)

        return (partial1 and anti2) or (anti1 and partial2)

    return cond


# ============================================== Case For Edge Types ====================================================== #
def is_anti_s_edge(t: FunctionalTree, ancestor_edge: Node) -> bool:
    return t._edge_types[ancestor_edge.split_indices] == "anti"


def is_full_s_edge(t: FunctionalTree, ancestor_edge: Node) -> bool:
    return t._edge_types[ancestor_edge.split_indices] == "full"


def is_partial_s_edge(t: FunctionalTree, ancestor_edge: Node) -> bool:
    return t._edge_types[ancestor_edge.split_indices] == "partial"


def is_none_edge(t: FunctionalTree, ancestor_edge: Node) -> bool:
    return t._edge_types[ancestor_edge.split_indices] == "none"


def algo5_partial_none_only_partial(t1):
    def cond(component):
        ancestor_edge1 = get_ancestor_edge(t1, component)

        partial1 = is_partial_s_edge(t1, ancestor_edge1)

        return partial1

    return cond


def algo5_partial_none_only_anti_sedge(t1):
    def cond(component):
        ancestor_edge1 = get_ancestor_edge(t1, component)

        anti1 = is_anti_s_edge(t1, ancestor_edge1)

        return anti1

    return cond


def algo5_is_none_s_edge(t1):
    def cond(component):
        ancestor_edge1 = get_ancestor_edge(t1, component)

        none_edge = is_none_edge(t1, ancestor_edge1)

        return none_edge

    return cond


def algo5_ancestor_is_partial_s_edge(t1):
    def cond(component):
        ancestor_edge1 = get_ancestor_edge(t1, component)

        none_edge = is_none_edge(t1, ancestor_edge1)

        return none_edge

    return cond


# ============================================== Case For Edge Types ====================================================== #
def case_full_full(
    sedge: Node,
    t1: FunctionalTree,
    t2: FunctionalTree,
):
    return algo1(sedge, t1, t2)


def case_full_none(
    sedge, t1: FunctionalTree, t2: FunctionalTree, original_tree_one, original_tree_two
):
    return algo1(sedge, t1, t2, original_tree_one, original_tree_two)


def case_partial_partial(
    sedge,
    t1: FunctionalTree,
    t2: FunctionalTree,
):
    c1 = calculate_component_set(t1, sedge)
    c2 = calculate_component_set(t2, sedge)

    c1 = filter_components_from_arms(algo5_partial_partial_cond(t1, t2), c1)
    c2 = filter_components_from_arms(algo5_partial_partial_cond(t2, t1), c2)

    c1 = filter_(lambda x: len(x) != 0, c1)
    c2 = filter_(lambda x: len(x) != 0, c2)

    c12 = cartesian(c1, c2)

    intersections = map2(intersect, c12)
    intersections = remove_empty_lists(intersections)

    symmetric_differences = map2(symm, c12)
    symmetric_differences = remove_empty_lists(symmetric_differences)

    voting_map = intersections + symmetric_differences
    voting_map_filtered = filter_(lambda x: x, voting_map)

    m: list[Component] = argmax(
        voting_map_filtered, lambda x: count(voting_map_filtered, x)
    )

    m = argmin(m, size)

    c = map1(remove_last_component_if_longer_than_one, m)

    c = reduce(union, c)

    return c


def case_partial_none(
    sedge,
    partial_sedge_subtree: FunctionalTree,
    none_s_edge_subtree: FunctionalTree,
):
    c1 = calculate_component_set(partial_sedge_subtree, sedge)
    c2 = calculate_component_set(none_s_edge_subtree, sedge)
    r = find_exact_max_intersection(c1, c2)
    return r


def algo1(
    sedge: Node,
    t1: FunctionalTree,
    t2: FunctionalTree,
) -> list[Component]:
    c1: list[ComponentSet] = calculate_component_set(t1, sedge)

    c2: list[ComponentSet] = calculate_component_set(t2, sedge)

    c12: list[tuple[ComponentSet, ComponentSet]] = cartesian(c1, c2)

    intersections: list[ComponentSet] = map2(cut, c12)

    voting_map = remove_empty_lists(intersections)

    symmetric_differences: list[ComponentSet] = map2(symm, c12)

    voting_map = remove_empty_lists(symmetric_differences)

    voting_map: list[ComponentSet] = intersections + symmetric_differences

    voting_map = remove_empty_lists(voting_map)

    m: list[Component] = argmax(voting_map, lambda x: count(voting_map, x))

    r: list[Component] = argmin(m, size)

    rr: list[Component] = reduce(union, r)

    if len({len(item) for item in r}) == 1:
        rr = next(iter(r))  # Preserve order without converting to set

    return rr


# ============================================== Algorithm 5 ====================================================== #
def algorithm_5_for_sedge(
    sedge,
    t1: FunctionalTree,
    t2: FunctionalTree,
    original_tree_one: Node,
    original_tree_two: Node,
):
    if is_full_s_edge(t1, sedge) and is_full_s_edge(t2, sedge):
        # debug_console.print("======== T1 is full T2 is full ======== ")
        return case_full_full(sedge, t1, t2)
    if is_full_s_edge(t1, sedge) and is_partial_s_edge(t2, sedge):
        # debug_console.print("========  T1 is full T2 is partial ========")
        return case_full_full(sedge, t1, t2)
    if is_partial_s_edge(t1, sedge) and is_full_s_edge(t2, sedge):
        # debug_console.print("======== T1 is partial T2 is full ========")
        return case_full_full(sedge, t2, t1)
    if is_partial_s_edge(t1, sedge) and is_partial_s_edge(t2, sedge):
        # debug_console.print("======== T1 is partial T2 is partial =======")
        return case_partial_partial(sedge, t1, t2)
    if is_partial_s_edge(t1, sedge) and is_none_edge(t2, sedge):
        # debug_console.print("======== T1 is partial T2 is none ========")
        return case_partial_none(
            sedge,
            t1,
            t2,
        )
    if is_none_edge(t1, sedge) and is_partial_s_edge(t2, sedge):
        # debug_console.print("======== T1 is none T2 is partial ========")
        return case_partial_none(sedge, t1, t2)
    if is_full_s_edge(t1, sedge):
        # debug_console.print("======== T1 is full ========")
        return case_full_full(sedge, t1, t2)
    if is_full_s_edge(t2, sedge):
        # debug_console.print("======== T2 is full ========")
        return case_full_full(
            sedge,
            t1,
            t2,
        )
    else:
        raise Exception(f"We forgot one case: {sedge}")


def algorithm_five(input_tree1: Node, input_tree2: Node, leaf_order: list[str]):
    """
    Runs 'algorithm five' on two trees, pruning iteratively based on discovered components.

    Args:
        input_tree1 (Node): The first input tree (root node).
        input_tree2 (Node): The second input tree (root node).
        leaf_order (list[str]): The list of leaf labels in a certain (sorted) order.

    Returns:
        list[int]: A list of unique components (encoded as integer indices) discovered by the algorithm.
    """

    (
        intermediate_trees_one,
        consensus_tree_one,
        consensus_tree_two,
        intermediate_trees_two,
    ) = interpolate_tree(input_tree1, input_tree2)

    # 1. Build functional trees
    func_tree1 = build_functional_tree(intermediate_trees_one)
    func_tree2 = build_functional_tree(intermediate_trees_two)

    # 2. Global store for discovered components
    global_components: list[int] = []

    # 3. Merge S-edges from both functional trees
    merged_sedges: List[ComponentSet] = merge_sedges(
        func_tree1._all_sedges, func_tree2._all_sedges
    )
    # 4. Initialize pruned trees
    pruned_original_tree1, pruned_original_tree2 = input_tree1, input_tree2

    # Track remaining leaves dynamically

    remaining_leaves = len(leaf_order)

    # 5. Loop until no more pruning is required
    while True:
        iteration_components = []
        iteration_taxa = set()  # Collect unique taxa to delete in this iteration

        for s_edge in merged_sedges:
            new_components = algorithm_5_for_sedge(
                s_edge,
                func_tree1,
                func_tree2,
                pruned_original_tree1,
                pruned_original_tree2,
            )

            iteration_components.extend(new_components)
            for comp in new_components:
                iteration_taxa.update(comp)
            global_components += new_components

        current_proposed_deletions = len(iteration_taxa)
        # Check if any components found AND remaining leaves after deletion > 3
        if iteration_components and (remaining_leaves - current_proposed_deletions) > 3:
            # Update remaining leaves count
            remaining_leaves -= current_proposed_deletions

            input_tree1.delete_taxa(indices_to_delete=iteration_taxa)
            input_tree2.delete_taxa(indices_to_delete=iteration_taxa)

            interpolated_pruned_tree1, interpolated_pruned_tree2 = (
                get_intermediate_trees(pruned_original_tree1, pruned_original_tree2)
            )

            # Rebuild functional trees with pruned trees
            func_tree1 = build_functional_tree(interpolated_pruned_tree1)
            func_tree2 = build_functional_tree(interpolated_pruned_tree2)

            # Update merged sedges for next iteration
            merged_sedges = merge_sedges(func_tree1._all_sedges, func_tree2._all_sedges)
        else:
            break  # Exit loop if no components or insufficient leaves remain

    return list(set(global_components))


def get_intermediate_trees(original_tree_one: Node, original_tree_two: Node):
    interpolated_trees = interpolate_tree(original_tree_one, original_tree_two)
    return interpolated_trees[0], interpolated_trees[3]
