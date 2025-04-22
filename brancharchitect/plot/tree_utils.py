"""
Tree utility functions for traversal and information extraction.

This module provides functions for working with tree structures, including traversal,
finding nodes, and computing tree properties.
"""

from collections import deque
from typing import List, Generator, Dict, Optional, Tuple, Union, Set
from brancharchitect.tree import Node
from brancharchitect.plot.paper_plot.paper_plot_constants import DEFAULT_NODE_LENGTH

ZERO_LENGTH_TOLERANCE = 1e-9


def is_leaf(node: Node) -> bool:
    """Check if a node is a leaf (has no children)."""
    return len(node.children) == 0


def get_node_length(node: Node) -> float:
    """Return node.length if not None, else default to 1.0 for visualization."""
    return node.length if (node.length is not None) else DEFAULT_NODE_LENGTH


def get_node_id(node: Node, raw: bool = False) -> str:
    """
    Get a consistent identifier for a node that can be matched across trees.
    
    Args:
        node: The node to generate an ID for
        raw: If True, returns the raw node name without any prefix for internal nodes.
             This is needed for compatibility with highlight options in notebooks.
        
    Returns:
        A string ID that uniquely identifies the node
    """
    # Special case: if node is None, return empty string
    if not node:
        return ""
        
    # For raw mode (used in highlighting), just return the node name
    if raw:
        return node.name if node.name else ""
    
    # Standard behavior: use name directly for leaves, add prefix for internal nodes
    if is_leaf(node):
        return node.name
    return f"internal-{node.name}" if node.name else "internal"


def normalize_edge_id(edge_id: Union[Tuple[str, str], Tuple[Node, Node]]) -> Tuple[str, str]:
    """
    Normalize an edge ID to ensure consistent formatting.
    
    Can accept either:
    - A tuple of (parent_node, child_node) Node objects
    - A tuple of (parent_id, child_id) strings
    
    Args:
        edge_id: Edge identifier as either (Node, Node) or (str, str)
        
    Returns:
        Normalized (parent_id, child_id) strings
    """
    if isinstance(edge_id[0], Node) and isinstance(edge_id[1], Node):
        # If we have Node objects, get their raw IDs (without internal- prefix)
        return (get_node_id(edge_id[0], raw=True), get_node_id(edge_id[1], raw=True))
    
    # Otherwise, assume the edge_id is already string IDs and keep them as is
    return edge_id


def prepare_highlight_edges(edges: Set[Tuple]) -> Set[Tuple[str, str]]:
    """
    Process a set of edge tuples to ensure they use the correct ID format.
    
    This function handles the case where the highlight edges specified in notebooks
    use the raw node names without the 'internal-' prefix.
    
    Args:
        edges: Set of edge tuples to normalize
        
    Returns:
        Set of edge tuples with normalized IDs
    """
    return {normalize_edge_id(edge) for edge in edges}


def get_order(root: Node) -> List[str]:
    """
    Return the list of leaf names in the order encountered,
    using consistent node identification.
    """
    leaves = []
    for node in traverse(root):
        if is_leaf(node):
            leaves.append(get_node_id(node))
    return leaves


def traverse(root: Node) -> Generator[Node, None, None]:
    """
    Breadth-first traversal of a tree.
    
    Args:
        root: The root node of the tree
        
    Yields:
        Each node in the tree in breadth-first order
    """
    if not root:
        return

    queue = deque([root])
    visited = set([root])

    while queue:
        node = queue.popleft()
        yield node

        for child in node.children:
            if child and child not in visited:
                visited.add(child)
                queue.append(child)


def tree_depth(node: Node, depth: int = 0) -> int:
    """
    Calculate the maximum depth of a tree.
    
    Args:
        node: The current node
        depth: The current depth (for recursion)
        
    Returns:
        The maximum depth of the tree
    """
    if not node or node.is_leaf():
        return depth
    return max((tree_depth(child, depth + 1) for child in node.children), default=depth)


def get_node_label(node: Node) -> str:
    """
    Get the display label for a node.
    
    Args:
        node: The node to get the label for
        
    Returns:
        The label for the node
    """
    if node is None:
        return ""
    return node.name if node.name else ""


def find_node(root: Node, target_id: str) -> Optional[Node]:
    """
    Find a node in the tree by its ID.
    
    Args:
        root: The root node of the tree
        target_id: The ID to search for
        
    Returns:
        The found node, or None if not found
    """
    for node in traverse(root):
        if get_node_id(node) == target_id:
            return node
    return None


def get_leaves(root: Node) -> List[Node]:
    """
    Get all leaf nodes in a tree.
    
    Args:
        root: The root node of the tree
        
    Returns:
        A list of all leaf nodes
    """
    if not root:
        return []
    return [node for node in traverse(root) if node.is_leaf()]


def collapse_zero_length_branches(root: Node) -> Node:
    """
    Create a deep copy of the tree and collapse zero-length branches into polytomies.

    Args:
        root: The root node of the tree

    Returns:
        A new tree with zero-length branches collapsed
    """
    if not root:
        return root

    # Create a deep copy of the tree
    try:
        if hasattr(root, "deep_copy"):
            tree_copy = root.deep_copy()
        else:
            # Fall back to standard library deepcopy
            import copy
            tree_copy = copy.deepcopy(root)
    except Exception as e:
        print(f"Warning: Could not deep copy tree: {e}. Using original tree.")
        tree_copy = root

    # Recursively collapse zero-length branches
    _collapse_zero_branches_recursive(tree_copy)

    return tree_copy


def _collapse_zero_branches_recursive(node: Node) -> None:
    """
    Recursively collapse zero-length internal branches.

    This is a post-order traversal - we process children first, then the current node.
    
    Args:
        node: The current node to process
    """
    if node.is_leaf():
        return

    # Process all children first (post-order)
    for child in list(node.children):
        _collapse_zero_branches_recursive(child)

    # Now check this node's children for collapsing
    new_children = []
    modified = False

    for child in node.children:
        # Get length (default to 0 if None or negative)
        length = child.length if child.length is not None and child.length > 0 else 0.0

        # If internal node with effectively zero length
        if child.is_internal() and length <= ZERO_LENGTH_TOLERANCE:
            # Add all grandchildren directly to current node
            for grandchild in child.children:
                grandchild.parent = node
                new_children.append(grandchild)
            modified = True
        else:
            # Keep this child
            new_children.append(child)

    # Update children list if modified
    if modified:
        node.children = new_children


def calculate_max_path_length(root: Node) -> float:
    """
    Calculate the maximum cumulative path length from root to any leaf.
    
    Args:
        root: The root node of the tree
        
    Returns:
        The maximum path length
    """
    if not root:
        return 0.0

    max_length = 0.0
    queue = deque([(root, 0.0)])
    visited = set()

    while queue:
        node, distance = queue.popleft()
        if node in visited:
            continue

        visited.add(node)

        # Update max distance if this is a leaf
        if node.is_leaf():
            max_length = max(max_length, distance)

        # Add all children to queue with updated distance
        for child in node.children:
            if child not in visited:
                child_length = (
                    child.length
                    if child.length is not None and child.length > 0
                    else 0.0
                )
                queue.append((child, distance + child_length))

    return max_length


def calculate_node_depths(root: Node, depth: int = 0, depth_dict: Optional[Dict[Node, int]] = None) -> Dict[Node, int]:
    """
    Calculate depth for each node in a tree.
    
    Args:
        root: The root node of the tree
        depth: Current depth (0 for root)
        depth_dict: Dictionary to store node -> depth mapping
        
    Returns:
        A dictionary mapping each node to its depth
    """
    if depth_dict is None:
        depth_dict = {}
    
    # Set the depth for this node
    depth_dict[root] = depth
    
    # Recursively set depths for children
    for child in getattr(root, 'children', []):
        calculate_node_depths(child, depth + 1, depth_dict)
    return depth_dict


def get_tree_size(root: Node) -> int:
    """
    Count the total number of nodes in a tree.
    
    Args:
        root: The root node of the tree
        
    Returns:
        The total number of nodes in the tree
    """
    count = 1  # Count the root
    for child in getattr(root, 'children', []):
        count += get_tree_size(child)
    return count


def calculate_all_node_depths(roots):
    """
    Calculate depths for all nodes across multiple roots.
    
    Args:
        roots: List of root nodes
        
    Returns:
        A dictionary mapping each node to its depth
    """
    all_node_depths = {}
    for root in roots:
        all_node_depths.update(calculate_node_depths(root))
    return all_node_depths


def inject_depth_information(roots, layouts, all_node_depths):
    """
    Inject depth information into layouts based on node depths.
    
    Args:
        roots: List of root nodes
        layouts: Layouts to update
        all_node_depths: Dictionary of node depths
        
    Returns:
        Updated layouts with depth information
    """
    for root in roots:
        for node in traverse(root):
            if node in all_node_depths:
                depth = all_node_depths[node]
                if node in layouts:
                    layouts[node]['depth'] = depth
    return layouts