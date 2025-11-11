import math
import ast

from typing import Optional, Union, List, Dict, Tuple, Any
from contextvars import ContextVar
from brancharchitect.tree import Node

# Parsing behavior flag (context-local, controlled by parse_newick)
_TREAT_ZERO_AS_EPSILON: ContextVar[bool] = ContextVar(
    "_TREAT_ZERO_AS_EPSILON", default=False
)


# ===================================================================
# 1. METADATA PROCESSING FUNCTIONS
# ===================================================================


def split_token(token: str) -> Tuple[str, Any]:
    """
    Split a token into name and value parts.
    Handles both "name=value" and "name:value" formats for metadata.

    Args:
        token: A string token in format "name=value" or "name:value"

    Returns:
        Tuple of (name, parsed_value) where parsed_value could be string, int, or float
    """
    # Handle both "=" and ":" separators
    if "=" in token:
        name, value = token.split("=", 1)
    elif ":" in token:
        name, value = token.split(":", 1)
    else:
        return token, True

    # Try to parse value as different types
    try:
        # First try as literal (handles strings in quotes, numbers, etc.)
        parsed_value = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        # If that fails, keep as string
        parsed_value = value

    return name, parsed_value


def flush_meta_buffer(meta_buffer: List[str], stack: List[Node]) -> None:
    """
    Process the metadata buffer and update the current node's metadata.

    Args:
        meta_buffer: List of characters that form the metadata content
        stack: The current stack of nodes being processed

    Returns:
        None - modifies the stack and meta_buffer in place
    """
    meta_string = "".join(meta_buffer)

    # Strip leading and trailing markers and whitespace
    meta_string = meta_string.strip()

    # Handle NHX format
    if meta_string.startswith("&&NHX:"):
        meta_string = meta_string[6:]  # Remove "&&NHX:" prefix
        # Split by colons and process each token for NHX
        tokens = meta_string.split(":")
    else:
        # Handle generic metadata format (key=value pairs separated by commas or spaces)
        # Replace common separators with commas for consistent parsing
        meta_string = meta_string.replace(";", ",").replace(" ", ",")
        tokens = meta_string.split(",")

    metadata: Dict[str, Any] = {}

    for token in tokens:
        if token.strip():  # Skip empty tokens
            name, value = split_token(token.strip())
            metadata[name] = value

    # Add metadata to current node
    if metadata and stack:
        if not hasattr(stack[-1], "values"):
            stack[-1].values = {}
        stack[-1].values.update(metadata)

    meta_buffer.clear()


# ===================================================================
# 2. BUFFER PROCESSING FUNCTIONS
# ===================================================================


def flush_character_buffer(buffer: List[str], stack: List[Node]) -> None:
    """
    Process the character buffer and assign the name to the current node.

    Args:
        buffer: List of characters to join and assign as node name
        stack: The current stack of nodes being processed

    Returns:
        None - modifies the stack and buffer in place
    """
    if not stack:
        # If stack is empty, we're probably between trees - just clear buffer
        buffer.clear()
        return

    # Only set name if buffer has content
    if buffer:
        buffer_value = "".join(buffer)
        stack[-1].name = buffer_value

    buffer.clear()


def parse_metadata(data: str) -> Dict[str, Any]:
    """
    Parse a metadata string into a dictionary.
    Handles both regular metadata and NHX format.

    Args:
        data: String containing metadata in format "key1=value1,key2=value2"
              or NHX format "&&NHX:key1=value1:key2=value2"

    Returns:
        Dictionary mapping keys to their parsed values
    """
    # Handle NHX format
    if data.startswith("&&NHX:"):
        # Remove the &&NHX: prefix
        nhx_data = data[6:]
        # Split by colons for NHX format
        token_strings = nhx_data.split(":")
        # Process each key=value pair
        result: Dict[str, Any] = {}
        for token in token_strings:
            if "=" in token:
                key, value_str = token.split("=", 1)  # Split on first = only
                try:
                    # Try to parse as number first
                    if "." in value_str:
                        value: Any = float(value_str)
                    else:
                        value = int(value_str)
                except ValueError:
                    # If not a number, keep as string
                    value = value_str
                result[key] = value
        return result
    else:
        # Handle regular metadata format
        token_strings = data.split(",")
        token_pairs = [split_token(token) for token in token_strings]
        return dict(token_pairs)


# ===================================================================
# 2. BUFFER PROCESSING FUNCTIONS
# ===================================================================


def flush_length_buffer(buffer: List[str], stack: List[Node]) -> None:
    """
    Process the length buffer and assign the branch length to the current node.

    Args:
        buffer: List of characters to join and parse as branch length
        stack: The current stack of nodes being processed

    Returns:
        None - modifies the stack and buffer in place

    Raises:
        ValueError: If the buffer content cannot be parsed as a valid float
    """
    if not stack:
        # If stack is empty, we're probably between trees - just clear buffer
        buffer.clear()
        return

    buffer_value = "".join(buffer).strip()

    # Treat various null-like tokens as a very small positive value to keep topology
    # Examples encountered in datasets: "null", "None", empty after colon
    null_like = {"", "null", "NULL", "none", "None"}
    if buffer_value in null_like:
        stack[-1].length = 0.000005
        buffer.clear()
        return

    try:
        # Convert buffer_value to a float
        parsed_number = float(buffer_value)
        # Handle special float values: treat inf/nan as tiny positive to preserve structure
        if math.isinf(parsed_number) or math.isnan(parsed_number):
            parsed_number = 0.000005
        # Optionally treat explicit zeros as epsilon to avoid premature collapsing from dataset zeros
        if parsed_number == 0.0 and _TREAT_ZERO_AS_EPSILON.get():
            parsed_number = 0.000005
        stack[-1].length = parsed_number
    except ValueError:
        # If parsing fails (e.g., non-numeric token), treat as tiny positive length
        stack[-1].length = 0.000005
    buffer.clear()


def flush_buffer(buffer: List[str], stack: List[Node], mode: str) -> None:
    """
    Process the accumulated buffer based on the current parsing mode.

    Args:
        buffer: List of characters accumulated during parsing
        stack: The current stack of nodes being processed
        mode: Current parsing mode ("character_reader" or "length_reader")

    Returns:
        None - modifies the stack and buffer in place
    """
    if mode == "character_reader":
        flush_character_buffer(buffer, stack)
    elif mode == "length_reader":
        flush_length_buffer(buffer, stack)


# ===================================================================
# 3. NODE STACK MANAGEMENT FUNCTIONS
# ===================================================================


def init_nodestack() -> List[Node]:
    """
    Initialize the node stack with a dummy root node.

    Returns:
        List containing a single root node
    """
    root = Node(name="", length=1, depth=0)
    return [root]


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
    parent = stack[-1]
    new_node = Node(
        length=default_length,
        depth=(parent.depth + 1 if parent.depth is not None else 1),
    )

    # Add new_node to the current top node's children
    parent.children.append(new_node)

    # Set the child's parent pointer, making the tree pointer-based
    new_node.parent = parent

    stack.append(new_node)
    return stack, buffer, mode


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


# ===================================================================
# 4. CORE PARSING FUNCTIONS
# ===================================================================


def _parse_newick(tokens: str, default_length: float) -> List[Node]:
    """
    Return a list of top-level Node trees from the token string.

    This is the low-level parsing function that processes character by character.

    Args:
        tokens: Raw Newick format string
        default_length: Default branch length for nodes without explicit lengths

    Returns:
        List of parsed Node trees
    """
    trees: List[Node] = []
    buffer: List[str] = []
    meta_buffer: List[str] = []
    mode: str = "character_reader"
    node_stack: List[Node] = init_nodestack()

    for index in range(len(tokens)):
        char = tokens[index]

        if char == "\n":
            continue

        elif char == "(":
            # Start a new internal node
            if len(node_stack) == 0:
                node_stack = init_nodestack()
            node_stack, buffer, mode = create_new_node(
                node_stack, buffer, mode, default_length
            )
            mode = "character_reader"

        elif char == ")":
            flush_buffer(buffer, node_stack, mode)
            # When closing a node, do not pop the root dummy node
            if len(node_stack) > 1:
                close_node(node_stack, buffer, mode)
            mode = "character_reader"

        elif char == "," and mode in ["character_reader", "length_reader"]:
            flush_buffer(buffer, node_stack, mode)
            # Only close node if not at the root dummy node
            if len(node_stack) > 1:
                close_node(node_stack, buffer, mode)
            node_stack, buffer, mode = create_new_node(
                node_stack, buffer, mode, default_length
            )
            mode = "character_reader"

        elif char == ":" and mode != "metadata_reader":
            flush_buffer(buffer, node_stack, mode)
            mode = "length_reader"

        elif char == "[":  # metadata
            # Always flush the buffer before entering metadata mode
            flush_buffer(buffer, node_stack, mode)
            mode = "metadata_reader"

        elif char == "]" and mode == "metadata_reader":
            flush_meta_buffer(meta_buffer, node_stack)
            mode = "character_reader"

        elif char == ";":
            if mode == "metadata_reader":
                meta_buffer.append(char)
            else:
                flush_buffer(buffer, node_stack, mode)
                # Close all open nodes to complete the current tree
                while len(node_stack) > 1:
                    close_node(node_stack, buffer, mode)
                if len(node_stack) == 1:
                    trees.append(node_stack.pop())

                # Reset parser state for the next tree
                node_stack = []
                buffer = []
                meta_buffer = []
                mode = "character_reader"

        elif mode == "metadata_reader":
            meta_buffer.append(char)
        else:
            buffer.append(char)

    if len(node_stack) > 0:
        flush_buffer(buffer, node_stack, mode)
        while len(node_stack) > 1:
            close_node(node_stack, buffer, mode)
        assert len(node_stack) == 1
        trees.append(node_stack.pop())

    return trees


# ===================================================================
# 5. PUBLIC API FUNCTIONS
# ===================================================================


def parse_newick(
    tokens: str,
    order: Optional[List[str]] = None,
    encoding: Optional[Dict[str, int]] = None,
    default_length: float = 1.0,
    force_list: bool = False,
    treat_zero_as_epsilon: bool = False,
) -> Union[Node, List[Node]]:
    """
    Parse a Newick string into a tree or list of trees.

    This is the main public API function for parsing Newick format strings.
    It handles post-processing such as setting up order, encoding, and tree properties.

    Args:
        tokens: Newick format string
        order: Optional order for taxa names
        encoding: Optional encoding mapping for taxa names
        default_length: Default branch length for nodes without explicit lengths
        force_list: Always return a list even for single trees

    Returns:
        Single Node or list of Nodes representing parsed tree(s)
    """
    token = _TREAT_ZERO_AS_EPSILON.set(bool(treat_zero_as_epsilon))
    try:
        trees: List[Node] = _parse_newick(tokens, default_length=default_length)
    finally:
        # Restore previous behavior to avoid leaking state across calls
        _TREAT_ZERO_AS_EPSILON.reset(token)

    if order is None:
        # If user didn't supply an order, gather from first tree
        order = list(trees[0].get_current_order())

    if encoding is None:
        encoding = {name: idx for idx, name in enumerate(order)}

    # Post-process trees with metadata and ordering
    for idx, tree in enumerate(trees):
        tree.list_index = idx
        tree.taxa_encoding = encoding
        tree.initialize_split_indices(encoding)
        tree.fix_child_order()

    if len(trees) == 1 and not force_list:
        return trees[0]
    return trees


def get_linear_order(tree: Node) -> List[str]:
    """
    Get the linear order of taxa (leaf names) from a tree.

    Args:
        tree: The root node of a tree

    Returns:
        List of leaf node names in linear order
    """
    leaves = tree.leaves
    return [leaf.name for leaf in leaves if leaf.name]


# ===================================================================
