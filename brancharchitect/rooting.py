from collections import deque
from typing import Tuple, List
from brancharchitect.tree import Node  # Import Node for type annotations


# --- Robust midpoint rooting implementation ---
def find_farthest_leaves(root: Node) -> Tuple[Node, Node, float]:
    """Find the two leaves in the tree that are farthest apart and return them and their distance."""
    leaves = [n for n in root.traverse() if n.is_leaf()]
    if not leaves:
        raise ValueError("Tree has no leaves")
    start = leaves[0]
    farthest, dist = _bfs_farthest(start)
    other, max_dist = _bfs_farthest(farthest)
    return farthest, other, max_dist


def _bfs_farthest(start: Node) -> Tuple[Node, float]:
    visited = set()
    queue = deque([(start, 0.0)])
    farthest, max_dist = start, 0.0
    while queue:
        node, dist = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        if node.is_leaf() and dist > max_dist:
            farthest, max_dist = node, dist
        neighbors = []
        if node.parent:
            neighbors.append((node.parent, node.length or 0.0))
        for child in node.children:
            neighbors.append((child, child.length or 0.0))
        for neighbor, edge_len in neighbors:
            if neighbor not in visited:
                queue.append((neighbor, dist + edge_len))
    return farthest, max_dist


def path_between(node1: Node, node2: Node) -> List[Tuple[Node, float]]:
    """Return the path (as list of (node, edge_length)) from node1 to node2."""
    parent = {node1: (None, 0.0)}
    queue = deque([node1])
    found = False
    while queue and not found:
        node = queue.popleft()
        for neighbor, edge_len in _neighbors_with_length(node):
            if neighbor not in parent:
                parent[neighbor] = (node, edge_len)
                queue.append(neighbor)
                if neighbor == node2:
                    found = True
                    break
    path = []
    node = node2
    while node != node1:
        prev, edge_len = parent[node]
        path.append((node, edge_len))
        node = prev
    path.append((node1, 0.0))
    path.reverse()
    return path


def _neighbors_with_length(node: Node):
    if node.parent:
        yield node.parent, node.length or 0.0
    for child in node.children:
        yield child, child.length or 0.0


def midpoint_root(tree: Node) -> Node:
    """
    Return a new tree rooted at the midpoint of the longest path between any two leaves.
    """
    tree = tree.deep_copy()
    L1, L2, max_dist = find_farthest_leaves(tree)
    path = path_between(L1, L2)
    half = max_dist / 2.0
    acc = 0.0
    for i in range(1, len(path)):
        prev_node, _ = path[i - 1]
        node, edge_len = path[i]
        if acc + edge_len == half:
            return reroot_at_node(node)
        elif acc + edge_len > half:
            dist_from_prev = half - acc
            dist_from_next = edge_len - dist_from_prev
            return insert_root_on_edge(prev_node, node, dist_from_prev, dist_from_next)
        acc += edge_len
    raise RuntimeError("Failed to find midpoint on path")


def reroot_at_node(node: Node) -> Node:
    return _flip_upward(node)


def insert_root_on_edge(node1: Node, node2: Node, len1: float, len2: float) -> Node:
    # Disconnect node2 from node1
    if node2.parent == node1:
        node1.children.remove(node2)
        node2.parent = None
    elif node1.parent == node2:
        node2.children.remove(node1)
        node1.parent = None
    else:
        raise ValueError("Nodes are not directly connected")
    root = type(node1)(name="MidpointRoot", length=None)
    node1.length = len1
    node2.length = len2
    root.children = [node1, node2]
    node1.parent = root
    node2.parent = root
    return root


def _flip_upward(new_root: Node) -> Node:
    current = new_root
    prev_node = None
    prev_length = 0.0
    while current.parent:
        parent = current.parent
        old_length = current.length or 0.0
        if parent in current.children:
            current.children.remove(parent)
        if current in parent.children:
            parent.children.remove(current)
        current.parent = prev_node
        current.length = prev_length
        if prev_node is not None:
            prev_node.children.append(current)
        prev_node = current
        prev_length = old_length
        current = parent
    if current is not None:
        current.parent = prev_node
        current.length = prev_length
        if prev_node is not None and current not in prev_node.children:
            prev_node.children.append(current)
    return new_root
