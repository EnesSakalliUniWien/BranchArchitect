from brancharchitect.tree import Node
from typing import List, Callable, Tuple, Dict
from itertools import pairwise


def robinson_foulds_distance(tree1: Node, tree2: Node) -> float:
    splits1, splits2 = tree1.to_splits(), tree2.to_splits()
    set1 = set(splits1)
    set2 = set(splits2)
    return len(set1 ^ set2) / 2


def relative_robinson_foulds_distance(tree1: Node, tree2: Node) -> float:
    splits1, splits2 = tree1.to_splits(), tree2.to_splits()
    set1 = set(splits1)
    set2 = set(splits2)

    total_unique_differences = len(set1 ^ set2)  # Symmetric difference
    total_unique_splits = len(set1 | set2)  # Union of both sets

    if(total_unique_splits == 0):
        return 0.0
    relative_difference = total_unique_differences / total_unique_splits

    return relative_difference

def weighted_robinson_foulds_distance(tree1: Node, tree2: Node) -> float:
    """
    Calculate the weighted Robinson-Foulds distance between two trees.

    Args:
        tree1 (Node): The first tree
        tree2 (Node): The second tree

    Returns:
        float: The weighted Robinson-Foulds distance between the two trees.
    """
    splits1 = tree1.to_weighted_splits()
    splits2 = tree2.to_weighted_splits()

    all_splits = set(splits1) | set(splits2)

    weighted_distance = sum(
        abs(splits1.get(split, 0) - splits2.get(split, 0)) for split in all_splits
    )

    return weighted_distance


def calculate_along_trajectory(
    trajectory: List[Node], distance_function: Callable[[Node, Node], float]
) -> List[float]:
    dists = [distance_function(tree1, tree2) for tree1, tree2 in pairwise(trajectory)]
    return dists


def calculate_matrix_distance(
    trajectory: List[Node], distance_function: Callable[[Node, Node], float]
) -> List[List[float]]:
    n = len(trajectory)
    distance_matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if (
                i != j
            ):  # Optionally check to avoid computing distance from a node to itself
                distance_matrix[i][j] = distance_function(trajectory[i], trajectory[j])
    return distance_matrix


### Pairwise Distances ###


# Function to compute depths and parents
def compute_depths_and_parents(node: Node, depth: float, depths: Dict[str, float]):
    if node.name:
        depths[node.name] = depth
    for child in node.children:
        compute_depths_and_parents(child, depth + child.length, depths)


# Function to compute Euler tour and RMQ for LCA
def euler_tour(
    node: Node,
    depth: float,
    euler: List[Node],
    depths: Dict[Node, float],
    first_occurrence: Dict[Node, int],
):
    first_occurrence[node] = len(euler)
    euler.append(node)
    depths[node] = depth
    for child in node.children:
        euler_tour(child, depth + child.length, euler, depths, first_occurrence)
        euler.append(node)  # Add parent again after visiting child


# Preprocess function for LCA using RMQ
def preprocess_LCA(root: Node):
    euler = []
    depths = {}
    first_occurrence = {}
    euler_tour(root, 0.0, euler, depths, first_occurrence)
    n = len(euler)
    log_table = [0] * (n + 1)
    for i in range(2, n + 1):
        log_table[i] = log_table[i // 2] + 1
    k = log_table[n] + 1
    dp = [[0] * n for _ in range(k)]
    for i in range(n):
        dp[0][i] = i
    for j in range(1, k):
        for i in range(n - (1 << j) + 1):
            left = dp[j - 1][i]
            right = dp[j - 1][i + (1 << (j - 1))]
            dp[j][i] = left if depths[euler[left]] < depths[euler[right]] else right
    return euler, depths, first_occurrence, dp, log_table


# Function to find LCA using RMQ
def find_LCA(u: Node, v: Node, euler, depths, first_occurrence, dp, log_table):
    l = first_occurrence[u]
    r = first_occurrence[v]
    if l > r:
        l, r = r, l
    j = log_table[r - l + 1]
    left = dp[j][l]
    right = dp[j][r - (1 << j) + 1]
    return euler[left] if depths[euler[left]] < depths[euler[right]] else euler[right]


# Function to compute pairwise distances using LCA
def compute_pairwise_leaf_distances(tree: Node) -> Dict[Tuple[str, str], float]:
    depths = {}
    compute_depths_and_parents(tree, 0.0, depths)
    leaves = [leaf.name for leaf in tree.get_leaves()]
    name_to_node = {leaf.name: leaf for leaf in tree.get_leaves()}

    euler, node_depths, first_occurrence, dp, log_table = preprocess_LCA(tree)

    distances = {}
    for i in range(len(leaves)):
        for j in range(i + 1, len(leaves)):
            leaf1 = name_to_node[leaves[i]]
            leaf2 = name_to_node[leaves[j]]

            lca = find_LCA(
                leaf1, leaf2, euler, node_depths, first_occurrence, dp, log_table
            )
            dist = depths[leaves[i]] + depths[leaves[j]] - 2 * node_depths[lca]
            distances[(leaves[i], leaves[j])] = dist

    return distances


# Optimized Path Difference Distance Function
def path_difference_distance(tree1: Node, tree2: Node) -> float:
    """
    Calculate the Path Difference distance between two trees.

    Args:
        tree1 (Node): The first tree.
        tree2 (Node): The second tree.

    Returns:
        float: The Path Difference distance between the two trees.
    """
    # Compute pairwise distances for both trees
    distances1 = compute_pairwise_leaf_distances(tree1)
    distances2 = compute_pairwise_leaf_distances(tree2)

    # Get all unique leaf pairs
    leaf_pairs = set(distances1.keys()) | set(distances2.keys())

    # Sum the absolute differences of the path lengths
    total_difference = 0.0
    for pair in leaf_pairs:
        d1 = distances1.get(pair, 0.0)
        d2 = distances2.get(pair, 0.0)
        total_difference += abs(d1 - d2)

    return total_difference
