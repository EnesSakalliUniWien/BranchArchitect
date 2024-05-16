from tree_stack_parser import (
    parse_square_brackets,
    Node,
    get_circular_order,
    set_inner_node_indices,
    set_inner_nodes_as_splits,
)
import json

def get_split_list(node: Node, split_list: list):
    for child in node.children:
        get_split_list(child, split_list)
    split_list.append(node.split_indices)


def interpolate_tree(pair_bracket_tokens_one: str, pair_bracket_tokens_two: str):
    tree_one = parse_square_brackets(pair_bracket_tokens_one)

    # We focus on the first order of the tree
    order_list = get_circular_order(tree_one)

    # set inner nodes names by the inner nodes names
    set_inner_node_indices(tree_one, order_list)

    # get the list of splits in the tree
    split_list_tree_one = []
    get_split_list(tree_one, split_list_tree_one)

    # get the list of splits in the tree
    tree_two = parse_square_brackets(pair_bracket_tokens_two)
    set_inner_node_indices(tree_two, order_list)

    # get the list of splits of the second tree
    split_list_tree_two = []
    get_split_list(tree_two, split_list_tree_two)

    intermediate_tree_one = tree_two.deep_copy()
    set_not_existent_splits_to_zero(intermediate_tree_one, split_list_tree_two)

    intermediate_tree_two = tree_two.deep_copy()
    set_not_existent_splits_to_zero(intermediate_tree_two, split_list_tree_one)

    consensus_tree_one = intermediate_tree_one.deep_copy()
    remove_zero_nodes(consensus_tree_one)

    consensus_tree_two = intermediate_tree_two.deep_copy()
    remove_zero_nodes(consensus_tree_two)


def interpolate_tree(tree_one: Node, tree_two: Node):
    # get the list of splits in the tree
    split_list_tree_one = []
    get_split_list(tree_one, split_list_tree_one)

    # get the list of splits of the second tree
    split_list_tree_two = []
    get_split_list(tree_two, split_list_tree_two)

    intermediate_tree_one = tree_one.deep_copy()
    set_not_existent_splits_to_zero(intermediate_tree_one, split_list_tree_one)

    intermediate_tree_two = tree_two.deep_copy()
    set_not_existent_splits_to_zero(intermediate_tree_two, split_list_tree_two)

    consensus_tree_one = intermediate_tree_one.deep_copy()
    remove_zero_nodes(consensus_tree_one)

    consensus_tree_two = intermediate_tree_two.deep_copy()
    remove_zero_nodes(consensus_tree_two)

    return (
        intermediate_tree_one,
        consensus_tree_one,
        consensus_tree_two,
        intermediate_tree_two,
    )


def set_not_existent_splits_to_zero(intermediate_tree, split_list):
    if not contains_split(split_list, intermediate_tree.split_indices):
        intermediate_tree.length = 0
    for child in intermediate_tree.children:
        set_not_existent_splits_to_zero(child, split_list)


def remove_zero_nodes(node):
    if node is None:
        return None
    # Recursively handle all children first
    non_zero_children = []
    for child in node.children:
        processed_child = remove_zero_nodes(child)
        if processed_child is not None:
            non_zero_children.append(processed_child)

    node.children = non_zero_children

    # If current node is zero, reattach its children to its parent
    if node.length == 0:
        if node.parent is not None:
            for child in node.children:
                node.parent.add_child(child)
            return None  # Indicate that this node should be removed
        else:
            # Handle case if root node is zero
            # You might want to handle this case differently
            return node.children[0] if node.children else None
    return node


def contains_split(list_of_lists, target_list):
    target_set = set(target_list)
    return any(target_set == set(lst) for lst in list_of_lists)


def get_split_list(node: Node, split_list: list):
    for child in node.children:
        get_split_list(child, split_list)
    split_list.append(node.split_indices)
    return split_list


def interpolate_adjacent_tree_pairs(tree_list) -> list[Node]:
    results = []
    # Set inner node indices based on circular order
    circular_order = get_circular_order(parse_square_brackets(tree_list[0]))    

    for i in range(len(tree_list) - 1):
        tree_one_repr = tree_list[i]
        tree_two_repr = tree_list[i + 1]

        tree_one = parse_square_brackets(tree_one_repr)
        tree_two = parse_square_brackets(tree_two_repr)
        
        set_inner_nodes_as_splits(tree_one, circular_order)
        set_inner_nodes_as_splits(tree_two, circular_order)

        # Interpolate trees and get intermediate and consensus trees
        (
            intermediate_tree_one,
            consensus_tree_one,
            consensus_tree_two,
            intermediate_tree_two,
        ) = interpolate_tree(tree_one, tree_two)

        # Add the sequence for this pair to the results
        if i == 0:
            # Add the first original tree at the beginning of the sequence
            results.append(tree_one)

        results.extend(
            [
                intermediate_tree_one,
                consensus_tree_one,
                consensus_tree_two,
                intermediate_tree_two,
                tree_two,
            ]
        )
        
        return results        


def serialize_tree_list_to_json(tree_list: list[Node]):
    serialized_tree_list = []
    for tree in tree_list:
        serialized_tree_list.append(tree.serialize_to_dict())
    return serialized_tree_list


def write_tree_dictionaries_to_json(tree_list: list[Node], file_name: str):
    serialized_tree_list = serialize_tree_list_to_json(tree_list)
    with open(file_name, "w") as f:
        f.write(json.dumps(serialized_tree_list))


if __name__ == "__main__":
    # Example usage
    tree_list = [
        # "(((A:1,B:1):1,(C:1,D:1):1):1,(O1:1,O2:1):1);",
        # "(((A:1,B:1,D:1):1,C:1):1,(O1:1,O2:1):1);",
        # Add more trees as needed
        "(((A,B),C),O);",
        "(((A,C),B),O);",        
    ]
    processed_tree_pairs = interpolate_adjacent_tree_pairs(tree_list)
    # Serialize the tree list to JSON
    serialized_tree_list = serialize_tree_list_to_json(processed_tree_pairs)
    write_tree_dictionaries_to_json(processed_tree_pairs, "tree_list.json")
