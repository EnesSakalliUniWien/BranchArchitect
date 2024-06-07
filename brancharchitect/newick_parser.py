from brancharchitect.node import Node, serialize_to_dict_iterative
import math
import json
import ast
from typing import Optional


def get_linear_order(node: Node):
    order_list = []
    if len(node.children) == 0:
        return [node.name]
    for child in node.children:
        order_list = order_list + get_linear_order(child)
    return order_list

def get_taxa_name_circular_order(node: Node):
    order_list = []
    if len(node.children) == 0:
        return [node.leaf_name]
    else:
        for child in node.children:
            order_list = order_list + get_taxa_name_circular_order(child)
    return order_list

### Metadata

def split_token(token):
    if '=' in token:
        name, value = token.split('=')
    elif ':' in token:
        name, value =  token.split(':')
    else:
        raise ValueError(f'Metadata Token has neither "=" nor ":", it is {token}')
    value = ast.literal_eval(value)
    return (name, value)


def parse_metadata(data: str):
    tokens = data.split(',')
    tokens = [split_token(token) for token in tokens]
    return dict(tokens)


def flush_meta_buffer(meta_buffer, stack):
    buffer_value = "".join(meta_buffer)
    meta_dict = parse_metadata(buffer_value)
    stack[-1].values = meta_dict
    meta_buffer.clear()


##### Tree Stack Parser #####

def flush_buffer(buffer, stack, mode):

    if mode == "character_reader":
        buffer_value = "".join(buffer)
        stack[-1].name = buffer_value
        buffer.clear()

    elif mode == "length_reader":
        buffer_value = "".join(buffer).strip()
        try:
            # Convert buffer_value to a float
            parsed_number = float(buffer_value)
            # Handle special float values if needed
            if math.isinf(parsed_number):
                raise ValueError("Parsed an invalid length inf")
            if math.isnan(parsed_number):
                raise ValueError("Parsed an invalid length NaN")
            stack[-1].length = parsed_number
        except ValueError as e:
            raise ValueError(f"Failed to parse '{buffer_value}' as a float: {str(e)}")
        buffer.clear()


def close_node(stack, buffer, mode):
    stack.pop()
    return stack, buffer, mode


def create_new_node(stack, buffer, mode, indices, default_length):
    new_node = Node(length=default_length, indices=indices)

    stack[-1].children.append(new_node)
    #new_node.parent = stack[-1]

    stack.append(new_node)
    return stack, buffer, mode

def init_nodestack():
    root = Node(indices=0, name='root', parent=None, length=1)
    return [root]

def parse_newick(tokens: str, order: Optional[list[str]]=None, default_length=1) -> Node:
    trees = _parse_newick(tokens, default_length=default_length)

    if order is None:
        order = get_linear_order(trees[0])

    for tree in trees:
        tree._initialize_split_indices(order)
        tree._order = order

    if len(trees) == 1:
        return trees[0]
    return trees


def _parse_newick(tokens: str, default_length) -> Node:
    trees = []
    buffer = []
    meta_buffer = []
    mode = "character_reader"
    node_stack = init_nodestack()

    for index in range(len(tokens)):
        char = tokens[index]
        if char == "(":
            if len(node_stack) == 0:
                node_stack = init_nodestack()
            node_stack, buffer, mode = create_new_node(node_stack, buffer, mode, index, default_length)
            mode = "character_reader"
        elif char == ")":
            flush_buffer(buffer, node_stack, mode)
            close_node(node_stack, buffer, mode)
            mode = "character_reader"
        elif char == "," and mode in ['character_reader', 'length_reader']:
            flush_buffer(buffer, node_stack, mode)
            close_node(node_stack, buffer, mode)
            node_stack, buffer, mode = create_new_node(node_stack, buffer, mode, index, default_length)
            mode = "character_reader"
        elif char == ":":
            flush_buffer(buffer, node_stack, mode)
            mode = "length_reader"
        elif char == "[":  # meta data
            mode = 'metadata_reader'
        elif char == "]" and mode == "metadata_reader":
            flush_meta_buffer(meta_buffer, node_stack)
            mode = 'character_reader'
        elif char == ";":
            if mode == 'metadata_reader':
                meta_buffer.append(char)
            else:
                flush_buffer(buffer, node_stack, mode)
                assert len(node_stack) == 1
                trees.append(node_stack.pop())
        elif mode == "metadata_reader":
            meta_buffer.append(char)
        else:
            buffer.append(char)

    if len(node_stack) > 0:
        flush_buffer(buffer, node_stack, mode)
        assert len(node_stack) == 1
        trees.append(node_stack.pop())

    return trees
