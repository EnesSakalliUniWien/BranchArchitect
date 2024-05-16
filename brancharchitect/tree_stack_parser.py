from brancharchitect.node import Node, serialize_to_dict_iterative
import json


##### Meta Value Buffer #####
def linear_flush_buffer(buffer):
    buffer_value = "".join(buffer)
    buffer.clear()
    return buffer_value


def parse_square_brackets_linear(tokens: str):
    tokens.strip()
    buffer = []
    meta_value_buffer = []
    mode = "skip"
    for index in range(0, len(tokens), 1):
        if tokens[index] == "[":
            mode = "READ"
            continue
        if tokens[index] == ",":
            flushed_buffer_entry = linear_flush_buffer(buffer)
            meta_value_buffer.append(flushed_buffer_entry)
            continue
        if tokens[index] == "]":
            flushed_buffer_entry = linear_flush_buffer(buffer)
            meta_value_buffer.append(flushed_buffer_entry)
            return meta_value_buffer, index + 1
        if mode == "READ":
            buffer.append(tokens[index])


def flush_meta_value_buffer(buffer, stack):
    buffer_value = "".join(buffer)
    stack[-1].values.append(buffer_value)
    buffer.clear()


##### Tree Stack Parser #####
def flush_buffer(buffer, stack, mode):
    if mode == "character_reader":
        buffer_value = "".join(buffer)
        stack[-1].name = buffer_value
        buffer.clear()
    if mode == "length_reader":
        buffer_value = "".join(buffer).strip()
        try:
            # Convert buffer_value to a float
            parsed_number = float(buffer_value)
            # Handle special float values if needed
            if (
                parsed_number == float("inf")
                or parsed_number == float("-inf")
                or parsed_number != parsed_number
            ):  # NaN check
                raise ValueError(
                    f"Buffer contains a special float value ('Inf', '-Inf', or 'NaN'): '{buffer_value}'"
                )
            stack[-1].length = parsed_number
        except ValueError as e:
            # Provide a detailed error message
            raise ValueError(f"Failed to parse '{buffer_value}' as a float: {str(e)}")
        buffer.clear()


def close_node(stack, buffer, mode):
    stack.pop()
    return stack, buffer, mode


def create_new_node(stack, buffer, mode, indices):
    new_node = Node()
    stack[-1].children.append(new_node)
    new_node.parent = stack[-1]
    stack.append(new_node)
    stack[-1].indices = indices
    return stack, buffer, mode


def parse_square_brackets(tokens: str) -> Node:
    tokens.strip()

    buffer = []
    mode = "character_reader"
    index = 0
    node_stack = [Node()]
    node_stack[-1].indices = index
    node_stack[-1].name = "root"
    node_stack[-1].parent = None
    node_stack[-1].length = 1

    for index in range(0, len(tokens), 1):
        if tokens[index] == "(":
            node_stack, buffer, mode = create_new_node(node_stack, buffer, mode, index)
            mode = "character_reader"
            continue
        if tokens[index] == ")":
            flush_buffer(buffer, node_stack, mode)
            close_node(node_stack, buffer, mode)
            mode = "character_reader"
            continue
        if tokens[index] == ",":
            flush_buffer(buffer, node_stack, mode)
            close_node(node_stack, buffer, mode)
            node_stack, buffer, mode = create_new_node(node_stack, buffer, mode, index)
            mode = "character_reader"
            continue
        if tokens[index] == ":":
            flush_buffer(buffer, node_stack, mode)
            mode = "length_reader"
            continue
        if tokens[index] == "[":  # meta data
            meta_values_buffer, meta_values_offset = parse_square_brackets_linear(
                tokens[index:]
            )
            index += meta_values_offset
            flush_meta_value_buffer(meta_values_buffer, node_stack)
        if mode == "character_reader":
            buffer.append(tokens[index])
        if mode == "length_reader":
            buffer.append(tokens[index])
    return node_stack[-1]


def set_inner_node_names(node):
    for child in node.children:
        set_inner_node_names(child)
    if not node.children:
        return [node.name]
    else:
        for child in node.children:
            node.name += child.name


def set_inner_node_indices(node: Node(), order_list):
    for child in node.children:
        set_inner_node_indices(child, order_list)
    if not node.children:
        node.split_indices = [order_list.index(node.name)]
    else:
        for child in node.children:
            node.split_indices += child.split_indices


def set_inner_nodes_as_splits(node, order_tuple):
    for child in node.children:
        set_inner_nodes_as_splits(child, order_tuple)  # Recursive call
    if not node.children:
        # Add the index of the node name if it's in the order_tuple
        node.leaf_name = node.name
        node.split_indices = (order_tuple.index(node.name),)
        node.name = node.split_indices
    else:
        node.name = ()
        node.split_indices = ()
        for child in node.children:
            node.split_indices += child.split_indices  # Concatenate tuples
            node.name += child.split_indices


def get_circular_order(node: Node):
    order_list = []
    if len(node.children) == 0:
        return [node.name]
    for child in node.children:
        order_list = order_list + get_circular_order(child)
    return order_list


def get_taxa_name_circular_order(node: Node):
    order_list = []
    if len(node.children) == 0:
        return [node.leaf_name]
    else:
        for child in node.children:
            order_list = order_list + get_taxa_name_circular_order(child)
    return order_list


# processed_tree_pairs contains the specified sequence for each adjacent pair
def transform_tree_from_file(file_name):
    newick_string = ""
    with open(file_name) as f:
        newick_string = f.readline()
    tree = parse_square_brackets(newick_string)
    serialized_tree = serialize_to_dict_iterative(tree)
    f = open(f"{file_name}.json", "w")
    f.write(json.dumps(serialized_tree))
    f.close()
    return serialized_tree


if __name__ == "__main__":
    # Example usage
    tree_list = [
        "(((A:1,B:1):1,(C:1,D:1):1):1,(O1:1,O2:1):1);",
        "(((A:1,B:1,D:1):1,C:1):1,(O1:1,O2:1):1);",
        # Add more trees as needed
    ]

    first_parsed_tree = parse_square_brackets(tree_list[0])
    # order = get_circular_order(first_parsed_tree)
    # set_inner_nodes_as_splits(first_parsed_tree, order)
    # print(first_parsed_tree.dict_to_json_string())
