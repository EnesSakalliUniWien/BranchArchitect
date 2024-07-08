from brancharchitect.jumping_taxa.elemental import merge_sedges, decode_indices_to_taxa
from brancharchitect.jumping_taxa.deletion_algorithm import delete_taxa
from brancharchitect.jumping_taxa.algorithm_one import find_jumping_taxa_algorithm_one
from brancharchitect.jumping_taxa.elemental import calculate_component_set
from logging import getLogger
from brancharchitect.jumping_taxa.functional_tree import (
    FunctionalTree,
    ComponentSet,
    Component,
    build_functional_tree,
)
from brancharchitect.jumping_taxa.elemental import (
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
)
from brancharchitect.jumping_taxa.tree_interpolation import (
    interpolate_adjacent_tree_pairs,
    interpolate_tree,
)
from brancharchitect.newick_parser import get_taxa_name_circular_order
from brancharchitect.tree import Node

logger = getLogger(__name__)


def print_component_map(component_set, sorted_nodes, title=None):
    if title:
        logger.info(title)
    logger.info(component_set)
    for component_set in component_set:
        components_converted = []
        for components in component_set:
            components_converted.append(
                [sorted_nodes[sub_component] for sub_component in components]
            )
        logger.info(components_converted)


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

# ============================================== Case For Edge Types ====================================================== #
def case_full_full(sedge: Node, t1: FunctionalTree, t2):
    return find_jumping_taxa_algorithm_one(sedge, t1, t2)


def case_full_none(sedge, t1, t2):
    return find_jumping_taxa_algorithm_one(sedge, t1, t2)


def case_partial_partial(sedge, t1, t2, sorted_nodes):
    c1 = calculate_component_set(t1, sedge)
    c2 = calculate_component_set(t2, sedge)

    print_component_map(c1, sorted_nodes, "C1")
    print_component_map(c2, sorted_nodes, "C2")

    cf1 = filter_components_from_arms(algo5_partial_partial_cond(t1, t2), c1)
    cf2 = filter_components_from_arms(algo5_partial_partial_cond(t2, t1), c2)

    print_component_map(cf1, sorted_nodes, "CF1")
    print_component_map(cf2, sorted_nodes, "CF2")

    cff1 = filter_(lambda x: len(x) != 0, cf1)
    cff2 = filter_(lambda x: len(x) != 0, cf2)

    print_component_map(cff1, sorted_nodes, "CFF1")
    print_component_map(cff2, sorted_nodes, "CFF2")

    c12 = cartesian(cff1, cff2)

    intersections = map2(intersect, c12)
    print_component_map(intersections, sorted_nodes, "Intersections")

    symmetric_differences = map2(symm, c12)

    print_component_map(symmetric_differences, sorted_nodes, "Symmetric Differences")

    voting_map = intersections + symmetric_differences

    voting_map_filtered = filter_(lambda x: x, voting_map)

    m: list[Component] = argmax(
        voting_map_filtered, lambda x: count(voting_map_filtered, x)
    )

    m = argmin(m, size)

    c = map1(remove_last_component_if_longer_than_one, m)

    c = reduce(union, c)

    return c


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


def case_partial_none(sedge, t1, t2, sorted_nodes):
    c1 = calculate_component_set(t1, sedge)
    c2 = calculate_component_set(t2, sedge)

    print_component_map(c1, sorted_nodes, "C1")
    print_component_map(c2, sorted_nodes, "C2")

    cf1_anti_s_edge = filter_components_from_arms(
        algo5_partial_none_only_anti_sedge(t1), c1
    )

    cf1_partial_s_edge = filter_components_from_arms(
        algo5_partial_none_only_partial(t1), c1
    )

    print_component_map(cf1_anti_s_edge, sorted_nodes, "CF1 Anti S-edge")

    print_component_map(cf1_partial_s_edge, sorted_nodes, "Partial S-edge")

    cf1_partial_s_edge = [reduce(union, cf1_partial_s_edge)]

    print_component_map(cf1_partial_s_edge, sorted_nodes, "Reduced partial s-edges")

    combined = cf1_partial_s_edge + cf1_anti_s_edge

    print_component_map(combined, sorted_nodes, "Combined")

    cf1 = argmin(combined, size)

    c = map1(remove_last_component_if_longer_than_one, cf1)

    c = reduce(union, c)

    return c


# ============================================== Algorithm 5 ====================================================== #


def algorithm_5_for_sedge(sedge, t1: FunctionalTree, t2: FunctionalTree, sorted_nodes):
    if is_full_s_edge(t1, sedge) and is_full_s_edge(t2, sedge):
        logger.info("Full Full")
        return case_full_full(sedge, t1, t2)

    if is_full_s_edge(t1, sedge) and is_partial_s_edge(t2, sedge):
        logger.info("Full Partial")
        return case_full_full(sedge, t1, t2)

    if is_partial_s_edge(t1, sedge) and is_full_s_edge(t2, sedge):
        logger.info("Partial Full")
        return case_full_full(sedge, t2, t1)

    if is_partial_s_edge(t1, sedge) and is_partial_s_edge(t2, sedge):
        logger.info("Partial Partial")

        return case_partial_partial(sedge, t1, t2, sorted_nodes)

    if is_partial_s_edge(t1, sedge) and is_none_edge(t2, sedge):
        logger.info("PARTIAL NONE")
        return case_partial_none(sedge, t1, t2, sorted_nodes)

    if is_none_edge(t1, sedge) and is_partial_s_edge(t2, sedge):
        logger.info("NONE PARTIAL")

        return case_partial_none(sedge, t2, t1, sorted_nodes)

    if is_full_s_edge(t1, sedge):
        return case_full_full(sedge, t1, t2)

    if is_full_s_edge(t2, sedge):
        return case_full_full(sedge, t1, t2)

    else:
        raise Exception(f"We forgot one case: {sedge}")


def algorithm_five(it1 : Node, it2: Node, sorted_nodes: list[str]):
    # Build functional trees from the intermediate trees
    t1 = build_functional_tree(it1)
    t2 = build_functional_tree(it2)

    # Initialize a list to store the global decoded results
    global_component_list: list[int] = []

    # Merge the S-edges from both trees
    all_s_edges = merge_sedges(t1._all_sedges, t2._all_sedges)

    # Initialize pruned trees
    p_it1 = it1
    p_it2 = it2

    while True:
        taxa = []
        component_indices = []

        # Iterate over all S-edges
        for s_edge in all_s_edges:
            
            # Execute Algorithm 5 for the current S-edge
            component_indices = algorithm_5_for_sedge(s_edge, t1, t2, sorted_nodes)
                                    
            global_component_list += component_indices
            
            # Translate taxa to indices
            taxa = list(set([y for x in component_indices for y in x]))
            
            # Append to global decoded result list
            global_component_list += component_indices
            
        # Stop condition for pruning
        if len(component_indices) > 0 and (len(sorted_nodes) - len(taxa) > 3):
            
            # Delete leaves and interpolate
            p_it1, p_it2, _ = delete_leave_and_interpolate(p_it1, p_it2, taxa)
            
            # Rebuild functional trees and S-edges after pruning            
            t1 = build_functional_tree(p_it1)
            t2 = build_functional_tree(p_it2)            
            
            all_s_edges = t1._all_sedges.union(t2._all_sedges)
            
        else:
            # Break the loop if no further pruning is required
            break

    # Return the unique set of global decoded results
    return list(set(global_component_list))

# ============================================== Pruning ====================================================== #


def delete_leave_and_interpolate(original_tree_one : Node, original_tree_two : Node, to_be_deleted_leaves=[]):
    pruned_tree_one = delete_taxa(original_tree_one, to_be_deleted_leaves)
    pruned_tree_two = delete_taxa(original_tree_two, to_be_deleted_leaves)

    interpolated_trees = interpolate_tree(pruned_tree_one, pruned_tree_two)
    circular_order = get_taxa_name_circular_order(interpolated_trees[0])

    return interpolated_trees[0], interpolated_trees[3], circular_order


# ============================================== Main ====================================================== #

if __name__ == "__main__":
    adjacent_tree_list = interpolate_adjacent_tree_pairs(
        [
            "(((A:1,B:1):1,(C:1,D:1):1):1,(O1:1,O2:1):1);",
            "(((A:1,B:1,D:1):1,C:1):1,(O1:1,O2:1):1);",
        ]
    )
    first_order_tree = adjacent_tree_list[0]
    circular_order = get_taxa_name_circular_order(first_order_tree)
    results = algorithm_five(
        adjacent_tree_list[1], adjacent_tree_list[4], circular_order
    )
