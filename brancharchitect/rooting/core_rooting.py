"""
Core rerooting implementation for phylogenetic trees.

This module provides fundamental rerooting operations including:
- Basic tree structure manipulation
- Core rerooting operations
- Midpoint rooting algorithms
- Simple node matching

Author: BranchArchitect Team
"""

from typing import Optional, Tuple, List
from collections import deque
from brancharchitect.tree import Node
from brancharchitect.elements.partition import Partition

# =============================================================================
# HELPER FUNCTIONS FOR TREE STRUCTURE MANIPULATION
# =============================================================================


def _collect_path_to_root(start_node: Node) -> List[Node]:
    """
    Collect all nodes from start_node up to the current root.

    Args:
        start_node: The node to start collecting from

    Returns:
        List of nodes from start_node to root (inclusive)
    """
    path: List[Node] = []
    node: Optional[Node] = start_node
    while node is not None:
        path.append(node)
        node = node.parent
    return path


def _flip_upward(node: Node, collapse_single_child: bool = False) -> Node:
    """
    Flip the tree structure upward from the given node to make it the new root.

    This function reorganizes the tree structure by traversing from the given node
    up to the current root, reversing parent-child relationships along the path.

    Args:
        node: The node that should become the new root
        collapse_single_child: Whether to collapse single-child nodes (used for midpoint rooting)

    Returns:
        The new root node (same as input node)
    """
    if node.parent is None:
        return node  # Already root

    # Collect path from node to current root
    path: List[Node] = _collect_path_to_root(node)

    # Process the path to flip parent-child relationships
    # We process pairs along the path, flipping each edge
    for i in range(len(path) - 1):
        child_node = path[i]
        parent_node = path[i + 1]

        # Store the length that will move with the flipped relationship
        old_edge_length = child_node.length

        # Remove child from parent's children list
        parent_node.children = [c for c in parent_node.children if c is not child_node]

        # Make parent a child of child (flip the relationship)
        if not any(c is parent_node for c in child_node.children):
            child_node.children.append(parent_node)
        parent_node.parent = child_node

        # The edge length moves with the flipped relationship
        parent_node.length = old_edge_length

    # The target node becomes the new root
    node.parent = None
    node.length = None

    # Special case: if the new root has only one child, collapse the root edge
    # This happens when the midpoint falls on an edge, not a node
    if collapse_single_child:
        while len(node.children) == 1:
            only_child = node.children[0]
            # Attach grandchildren to the root
            node.children = only_child.children
            for gc in node.children:
                gc.parent = node
            # If the child had a length, add it to the root's length (should be None at root)
            # But for midpoint rooting, the root's length should remain None
            # Remove the collapsed child
            only_child.parent = None
            only_child.children = []

    return node


# =============================================================================
# CORE REROOTING OPERATIONS
# =============================================================================


def find_best_matching_node(target_partition: Partition, root: Node) -> Optional[Node]:
    """
    Returns the node in root whose Partition has the largest overlap with target_partition.
    Uses bitmask operations for efficient set intersection and early termination for perfect matches.

    Args:
        target_partition: The partition to match against
        root: Root of the tree to search in

    Returns:
        Node with the largest overlap, or None if no overlap found
    """
    if not target_partition or not target_partition.bitmask:
        return None

    target_bitmask = target_partition.bitmask
    best_node: Optional[Node] = None
    best_overlap = 0

    for node in root.traverse():
        if hasattr(node, "split_indices") and node.split_indices:
            if hasattr(node.split_indices, "bitmask") and node.split_indices.bitmask:
                node_bitmask = node.split_indices.bitmask

                # Check for exact match first
                if node_bitmask == target_bitmask:
                    return node

                # Use bitwise AND for intersection
                overlap = bin(target_bitmask & node_bitmask).count("1")

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_node = node

    return best_node


def simple_reroot(tree: Node, target_node: Node) -> Node:
    """
    Simple rerooting strategy that reroots the tree at the target node.

    Args:
        tree: Tree to reroot (actually unused since we reroot at target_node)
        target_node: Node to reroot at

    Returns:
        Rerooted tree with target_node as the new root
    """
    return _flip_upward(target_node)


def reroot_at_node(node: Node) -> Node:
    """
    Reroot the tree at the specified node.

    Args:
        node: The node to reroot at

    Returns:
        The new root node
    """
    return _flip_upward(node)


# =============================================================================
# MIDPOINT ROOTING
# =============================================================================


def _collect_all_leaves_bfs(root: Node) -> List[Node]:
    """
    Collect all leaves in the tree using BFS traversal.

    Args:
        root: Root node of the tree

    Returns:
        List of all leaf nodes
    """
    leaves: List[Node] = []
    queue = deque([root])

    while queue:
        node = queue.popleft()
        if not node.children:  # Leaf node
            leaves.append(node)
        else:
            queue.extend(node.children)

    return leaves


def _get_node_neighbors_with_distances(node: Node) -> List[Tuple[Node, float]]:
    """
    Get all neighboring nodes with their edge distances.

    Args:
        node: The node to get neighbors for

    Returns:
        List of (neighbor_node, distance) tuples
    """
    neighbors: List[Tuple[Node, float]] = []

    # Add parent if exists
    if node.parent:
        distance = node.length if node.length is not None else 0.0
        neighbors.append((node.parent, distance))

    # Add children
    for child in node.children:
        distance: float = child.length if child.length is not None else 0.0
        neighbors.append((child, distance))

    return neighbors


def _bfs_farthest(start_node: Node) -> Tuple[Node, float]:
    """
    Find the farthest node from start_node using BFS.

    Args:
        start_node: Node to start BFS from

    Returns:
        Tuple of (farthest_node, distance_to_farthest)
    """
    queue = deque([(start_node, 0.0)])
    visited = {start_node}
    farthest_node = start_node
    max_distance = 0.0

    while queue:
        current_node, current_distance = queue.popleft()

        if current_distance > max_distance:
            max_distance: float = current_distance
            farthest_node: Node = current_node

        # Explore neighbors
        for neighbor, edge_distance in _get_node_neighbors_with_distances(current_node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, current_distance + edge_distance))

    return farthest_node, max_distance


def _bfs_farthest_leaf(start_node: Node) -> Tuple[Node, float]:
    """
    Find the farthest LEAF from start_node using BFS.
    Excludes the start_node itself from consideration.

    Args:
        start_node: Node to start BFS from

    Returns:
        Tuple of (farthest_leaf, distance_to_farthest)
    """
    queue = deque([(start_node, 0.0)])
    visited = {start_node}
    farthest_leaf = None
    max_distance = -1.0  # Initialize to -1 so any leaf gets selected

    while queue:
        current_node, current_distance = queue.popleft()

        # Only consider leaf nodes for the result, but exclude the start node
        if not current_node.children and current_node is not start_node:
            if current_distance > max_distance:
                max_distance = current_distance
                farthest_leaf = current_node

        # Explore neighbors
        for neighbor, edge_distance in _get_node_neighbors_with_distances(current_node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, current_distance + edge_distance))

    # Fallback: if no leaf found, return the start node
    if farthest_leaf is None:
        farthest_leaf = start_node
        max_distance = 0.0

    return farthest_leaf, max_distance


def find_farthest_leaves(root: Node) -> Tuple[Node, Node, float]:
    """
    Find the two leaves that are farthest apart in the tree.
    
    Brute force approach: calculate distance between all pairs of leaves
    and return the pair with maximum distance.

    Args:
        root: Root of the tree

    Returns:
        Tuple of (leaf1, leaf2, distance_between_them)
    """
    # Collect all leaves
    leaves: List[Node] = _collect_all_leaves_bfs(root)

    if len(leaves) < 2:
        raise ValueError("Tree must have at least 2 leaves for midpoint rooting")

    max_distance = 0.0
    farthest_pair = (leaves[0], leaves[1])

    # Check all pairs of leaves
    for i in range(len(leaves)):
        for j in range(i + 1, len(leaves)):
            leaf1, leaf2 = leaves[i], leaves[j]
            
            # Calculate distance between this pair
            path_edges = path_between(leaf1, leaf2)
            distance = sum(weight for _, weight in path_edges)
            
            if distance > max_distance:
                max_distance = distance
                farthest_pair = (leaf1, leaf2)

    return farthest_pair[0], farthest_pair[1], max_distance


def path_between(node1: Node, node2: Node) -> List[Tuple[Node, float]]:
    """
    Find the path between two nodes in the tree.

    Args:
        node1: First node
        node2: Second node

    Returns:
        List of (node, edge_weight) tuples representing the path.
        The first tuple has edge_weight=0 for the starting node.
    """

    # Use the same algorithm as distance_between_leaves but return path with weights
    def path_to_root(leaf: Node) -> List[Node]:
        path = []
        node = leaf
        while node is not None:
            path.append(node)
            node = node.parent
        return path[::-1]  # Reverse to get root-to-leaf order

    path1: List[Node] = path_to_root(node1)
    path2: List[Node] = path_to_root(node2)

    # Find lowest common ancestor (LCA)
    min_len = min(len(path1), len(path2))
    lca_index = 0
    for i in range(min_len):
        if path1[i] is not path2[i]:  # Use identity comparison
            break
        lca_index = i

    # Build result path with weights
    result = []

    # Path from node1 up to (but not including) LCA
    current: Node = node1
    result.append((current, 0.0))  # Start node, weight 0
    while current is not path1[lca_index]:
        parent = current.parent
        weight = current.length if current.length is not None else 0.0
        result.append((parent, weight))
        current = parent

    # Now at LCA (already added if node1 != LCA)
    # Path from LCA down to node2 (excluding LCA)
    # Find the path from LCA to node2
    lca = path1[lca_index]
    down_path = []
    current = node2
    while current is not lca:
        down_path.append(current)
        current = current.parent
    down_path.reverse()
    for node in down_path:
        weight = node.length if node.length is not None else 0.0
        result.append((node, weight))

    return result


def _find_midpoint_on_path(leaf1: Node, leaf2: Node, total_distance: float) -> Node:
    """
    Find the midpoint node on the path between two leaves.

    Args:
        leaf1: First leaf
        leaf2: Second leaf
        total_distance: Total distance between leaves

    Returns:
        Node closest to the midpoint
    """
    target_distance = total_distance / 2.0
    path_edges = path_between(leaf1, leaf2)

    current_distance = 0.0
    for node, edge_weight in path_edges:
        if current_distance + edge_weight >= target_distance:
            # Midpoint is on this edge or at this node
            return node

        current_distance += edge_weight

    # If we reach here, return the last node
    return leaf2


def midpoint_root(tree: Node) -> Node:
    """
    Reroot the tree at its midpoint (center of the longest path).

    Args:
        tree: Root of the tree to reroot

    Returns:
        The rerooted tree
    """
    # Find the two farthest leaves
    leaf1, leaf2, total_distance = find_farthest_leaves(tree)

    # Find midpoint on the path between them
    midpoint_node: Node = _find_midpoint_on_path(leaf1, leaf2, total_distance)

    # Reroot at the midpoint
    return _flip_upward(midpoint_node, collapse_single_child=True)
