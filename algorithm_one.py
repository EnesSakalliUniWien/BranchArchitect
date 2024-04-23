from tree_stack_parser import Node, get_taxa_name_circular_order
from tree_interpolation import interpolate_adjacent_tree_pairs
from functional_tree import Component, ComponentSet
from functional_tree import FunctionalTree, build_functional_tree
from topology_change_algorithm import calculate_component_set


from topology_change_algorithm import (
    intersect,
    symm,
    union,
    remove_empty_lists,
    count,
    size,
    argmax,
    argmin,
    reduce,
    map2,
    decode_indices_to_taxa,
    cartesian,
)


def find_jumping_taxa_algorithm_one(
    s_edge: Node, t1: FunctionalTree, t2: FunctionalTree
) -> list[Component]:
    # Calculate the component sets for each tree with respect to the given S-edge
    c1: list[ComponentSet] = calculate_component_set(t1, s_edge)
    c2: list[ComponentSet] = calculate_component_set(t2, s_edge)

    # Generate cartesian product of component sets from both trees
    c12: list[tuple[ComponentSet, ComponentSet]] = cartesian(c1, c2)

    # Calculate the intersections of the component sets
    intersections: list[ComponentSet] = map2(intersect, c12)  # type: ignore

    # Calculate the symmetric differences of the component sets
    symmetric_differences: list[ComponentSet] = map2(symm, c12)  # type: ignore

    # Combine intersections and symmetric differences into a voting map
    voting_map: list[ComponentSet] = intersections + symmetric_differences

    # Remove empty lists from the voting map
    voting_map = remove_empty_lists(voting_map)

    # Find the components with the maximum occurrence in the voting map
    m: list[Component] = argmax(voting_map, lambda x: count(voting_map, x))

    # Identify the components with the minimum size
    r: list[Component] = argmin(m, size)

    # Reduce the components to a union set
    rr: list[Component] = reduce(union, r)  # type: ignore

    return rr


def algorithm_one(tree_list: list[Node], circular_order: list[str]):
    # Selecting two intermediate trees from the tree list
    intermediate_tree_one: Node = tree_list[1]
    intermediate_tree_two: Node = tree_list[4]

    # Building functional trees from the intermediate trees
    functional_tree_one = build_functional_tree(intermediate_tree_one)
    functional_tree_two = build_functional_tree(intermediate_tree_two)

    # Initializing a list to store global decode results
    global_decode_result_list = []

    # Merging S-edges from both functional trees
    all_s_edges = set(functional_tree_one._all_sedges + functional_tree_two._all_sedges)

    # Creating a map to keep track of taxa jumping
    taxa_jumping_map = {leave: 0 for leave in circular_order}

    # Iterating over all S-edges
    for s_edge in all_s_edges:
        # Finding jumping taxa for each S-edge
        jumping_taxa = find_jumping_taxa_algorithm_one(
            s_edge, functional_tree_one, functional_tree_two
        )

        # Decoding taxa indices to actual taxa names
        taxa = list([y for x in jumping_taxa for y in x])

        decoded_list = decode_indices_to_taxa(taxa, circular_order)

        # Updating the taxa jumping map for each leaf
        for leave in decoded_list:
            taxa_jumping_map[leave] += 1

        # Adding taxa with jumps to the global decode result list
        global_decode_result_list += [k for k, v in taxa_jumping_map.items() if v > 0]

    # Returning the unique set of global decode results
    return set(global_decode_result_list)


if __name__ == "__main__":
    adjacent_tree_list = interpolate_adjacent_tree_pairs(
        [
            "(((A:1,B:1):1,(C:1,D:1):1):1,(O1:1,O2:1):1);",
            "(((A:1,B:1,D:1):1,C:1):1,(O1:1,O2:1):1);",
        ]
    )
    first_order_tree = adjacent_tree_list[0]
    circular_order = get_taxa_name_circular_order(adjacent_tree_list[0])
    results = algorithm_one(adjacent_tree_list, circular_order)
