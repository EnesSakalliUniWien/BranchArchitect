from brancharchitect.node import Node, serialize_to_dict_iterative
import math
import json
import ast


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


def create_new_node(stack, buffer, mode, indices):
    new_node = Node()
    stack[-1].children.append(new_node)
    new_node.parent = stack[-1]
    stack.append(new_node)
    stack[-1].indices = indices
    return stack, buffer, mode

def init_nodestack():
    root = Node()
    root.indices = 0
    root.name = 'root'
    root.parent = None
    root.length = None

    return [root]

def parse_newick(tokens: str) -> Node:
    buffer = []
    meta_buffer = []
    mode = "character_reader"
    node_stack = init_nodestack()

    for index in range(len(tokens)):
        char = tokens[index]
        if char == "(":
            node_stack, buffer, mode = create_new_node(node_stack, buffer, mode, index)
            mode = "character_reader"
        elif char == ")":
            flush_buffer(buffer, node_stack, mode)
            close_node(node_stack, buffer, mode)
            mode = "character_reader"
        elif char == "," and mode in ['character_reader', 'length_reader']:
            flush_buffer(buffer, node_stack, mode)
            close_node(node_stack, buffer, mode)
            node_stack, buffer, mode = create_new_node(node_stack, buffer, mode, index)
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
                break
        elif mode == "metadata_reader":
            meta_buffer.append(char)
        else:
            buffer.append(char)
    return node_stack[-1]
