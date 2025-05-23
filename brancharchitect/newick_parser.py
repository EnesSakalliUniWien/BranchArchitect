import math
import ast
from typing import Optional, Union, List, Dict, Tuple, Any
from brancharchitect.tree import Node


### Metadata
def split_token(token: str) -> Tuple[str, Any]:
    """
    Split a metadata token into name and value parts.

    Args:
        token: String containing a name-value pair separated by "=" or ":"

    Returns:
        A tuple containing (name, value) where value is parsed using ast.literal_eval

    Raises:
        ValueError: If the token doesn't contain either "=" or ":" as a separator
    """
    name: str = ""
    value_str: str = ""
    if "=" in token:
        name = token.split("=")[0]
        value_str = token.split("=")[1]
    elif ":" in token:
        name = token.split(":")[0]
        value_str = token.split(":")[1]
    else:
        raise ValueError(f'Metadata Token has neither "=" nor ":", it is {token}')
    value: Any = ast.literal_eval(value_str)
    return (name, value)


def parse_metadata(data: str) -> Dict[str, Any]:
    """
    Parse a metadata string into a dictionary.

    Args:
        data: String containing metadata in format "key1=value1,key2=value2"

    Returns:
        Dictionary mapping keys to their parsed values
    """
    token_strings = data.split(",")
    token_pairs = [split_token(token) for token in token_strings]
    return dict(token_pairs)


def flush_meta_buffer(meta_buffer: List[str], stack: List[Node]) -> None:
    """
    Process the accumulated metadata buffer and apply it to the current node.

    Args:
        meta_buffer: A list of characters containing metadata information
        stack: The current stack of nodes being processed

    Returns:
        None - modifies the stack and meta_buffer in place
    """
    buffer_value = "".join(meta_buffer)
    meta_dict = parse_metadata(buffer_value)
    stack[-1].values = meta_dict
    meta_buffer.clear()


##### Tree Stack Parser #####


def flush_buffer(buffer: list[str], stack: list[Node], mode: str):

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


def close_node(
    stack: List[Node], buffer: List[str], mode: str
) -> Tuple[List[Node], List[str], str]:
    """
    Close the current node by removing it from the stack.

    Args:
        stack: The current stack of nodes being processed
        buffer: The current character buffer
        mode: The current parsing mode

    Returns:
        A tuple of (stack, buffer, mode) with the node popped from the stack
    """
    stack.pop()
    return stack, buffer, mode


def create_new_node(
    stack: List[Node], buffer: List[str], mode: str, default_length: float
) -> Tuple[List[Node], List[str], str]:
    """
    Create a new node and add it to the tree structure.

    Args:
        stack: The current stack of nodes being processed
        buffer: The current character buffer
        mode: The current parsing mode
        default_length: The default branch length to use for new nodes

    Returns:
        A tuple of (stack, buffer, mode) with the new node added to the stack
    """
    new_node = Node(length=default_length)

    # Add new_node to the current top node's children
    stack[-1].children.append(new_node)

    # Set the child's parent pointer, making the tree pointer-based
    new_node.parent = stack[-1]

    stack.append(new_node)
    return stack, buffer, mode


def init_nodestack() -> List[Node]:
    # We still create a dummy 'root' node at the top
    root = Node(name="root", length=1)
    return [root]


def parse_newick(
    tokens: str,
    order: Optional[List[str]] = None,
    encoding: Optional[Dict[str, int]] = None,
    default_length: float = 1.0,
    force_list: bool = False,
) -> Union[Node, List[Node]]:
    """
    Same parser logic as before, but now each node's .parent
    is set in create_new_node, making the tree pointer-based.
    """
    trees: List[Node] = _parse_newick(tokens, default_length=default_length)

    if order is None:
        # If user didn't supply an order, gather from first tree
        order = list(trees[0].get_current_order())

    if encoding is None:
        encoding = {name: idx for idx, name in enumerate(order)}

    for idx, tree in enumerate(trees):
        tree._list_index = idx
        tree._encoding = encoding
        tree._order = order
        tree._initialize_split_indices(encoding)
        tree._fix_child_order()

    if len(trees) == 1 and not force_list:
        return trees[0]
    return trees


def _parse_newick(tokens: str, default_length: float) -> List[Node]:
    """
    Return a list of top-level Node trees from the token string.
    """
    trees: list[Node] = []
    buffer: list[str] = []
    meta_buffer: list[str] = []
    mode: str = "character_reader"
    node_stack: list[Node] = init_nodestack()

    for index in range(len(tokens)):
        
        char = tokens[index]

        if char == "\n":
            continue

        elif char == "(":

            if len(node_stack) == 0:
                node_stack = init_nodestack()

            node_stack, buffer, mode = create_new_node(
                node_stack, buffer, mode, default_length
            )

            mode = "character_reader"

        elif char == ")":

            flush_buffer(buffer, node_stack, mode)

            close_node(node_stack, buffer, mode)

            mode = "character_reader"

        elif char == "," and mode in ["character_reader", "length_reader"]:
            
            flush_buffer(buffer, node_stack, mode)

            close_node(node_stack, buffer, mode)

            node_stack, buffer, mode = create_new_node(
                node_stack, buffer, mode, default_length
            )
            
            mode = "character_reader"
            
        elif char == ":":
            
            flush_buffer(buffer, node_stack, mode)
            
            mode = "length_reader"
            
        elif char == "[":  # meta data
            mode = "metadata_reader"
            
        elif char == "]" and mode == "metadata_reader":
            
            flush_meta_buffer(meta_buffer, node_stack)
            
            mode = "character_reader"
            
        elif char == ";":
            if mode == "metadata_reader":
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
