from typing import List
from brancharchitect.tree import Node
import numpy as np
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh

# --- Spectral (Fiedler vector) ordering utilities ---

# Memoization cache for leaf distance matrices
_leaf_distance_matrix_cache = {}


# Find paths from root to each leaf
def path_to_root(leaf: Node) -> List[Node]:
    path = []
    node = leaf
    while node is not None:
        path.append(node)
        node = node.parent
    return path[::-1]


def distance_between_leaves(tree: Node, leaf1: Node, leaf2: Node) -> float:
    """
    Compute the path length between two leaves in the tree.
    This implementation assumes unique leaf names and that each node has a 'length' attribute.
    """

    # from root to leaf

    path1 = path_to_root(leaf1)
    path2 = path_to_root(leaf2)
    # Find lowest common ancestor (LCA)
    min_len = min(len(path1), len(path2))
    lca_index = 0
    for i in range(min_len):
        if path1[i] != path2[i]:
            break
        lca_index = i
    # Path from leaf1 to LCA (excluding LCA)
    dist = 0.0
    for node in path1[lca_index + 1 :]:
        if node.length is not None:
            dist += node.length
    for node in path2[lca_index + 1 :]:
        if node.length is not None:
            dist += node.length
    return dist


def compute_leaf_distance_matrix(tree: Node) -> tuple:
    """
    Compute a pairwise distance matrix between all leaves in the tree.
    Returns (dist_matrix, leaf_names).
    Uses memoization to cache results for identical trees (by id).
    """
    cache_key = id(tree)
    if cache_key in _leaf_distance_matrix_cache:
        return _leaf_distance_matrix_cache[cache_key]
    leaves: List[Node] = tree.get_leaves()
    n = len(leaves)
    dist_matrix = np.zeros((n, n))
    for i, leaf1 in enumerate(leaves):
        for j, leaf2 in enumerate(leaves):
            if i < j:
                dist: float = distance_between_leaves(tree, leaf1, leaf2)
                dist_matrix[i, j] = dist_matrix[j, i] = dist
    result = (dist_matrix, [leaf.name for leaf in leaves])
    _leaf_distance_matrix_cache[cache_key] = result
    return result


def fiedler_ordering_for_tree(tree: Node) -> tuple:
    """
    Compute the Fiedler vector ordering for a single tree and reorder its nodes accordingly.
    Returns (ordering, fiedler_scores).
    """
    dist_matrix, leaf_names = compute_leaf_distance_matrix(tree)
    L = laplacian(dist_matrix, normed=True)
    eigvals, eigvecs = eigh(L)
    fiedler_vec = eigvecs[:, 1]
    fiedler_scores = dict(zip(leaf_names, fiedler_vec))
    ordering = [name for _, name in sorted(zip(fiedler_vec, leaf_names))]
    order_internal_nodes_by_fiedler(tree, fiedler_scores)
    return ordering, fiedler_scores


def order_internal_nodes_by_fiedler(node: Node, fiedler_scores: dict):
    """
    Recursively reorder the children of each internal node according to the minimum Fiedler value
    among their descendant leaves.
    """
    if node.is_leaf():
        return

    def child_score(child):
        return min(fiedler_scores[leaf.name] for leaf in child.get_leaves())

    node.children.sort(key=child_score)
    for child in node.children:
        order_internal_nodes_by_fiedler(child, fiedler_scores)


def consensus_distance_matrix(trees: List[Node]) -> tuple:
    """
    Compute a consensus (average) leaf-leaf distance matrix for a list of trees.
    Ensures matrices are aligned by leaf names before averaging.
    Returns (consensus_matrix, canonical_leaf_names).
    """
    if not trees:
        return np.array([]), []

    all_leaf_names = set()
    individual_matrices_info = []

    for tree in trees:
        dist_matrix, current_leaf_names = compute_leaf_distance_matrix(tree)
        if not current_leaf_names:  # Skip trees with no leaves or handle as error
            continue
        all_leaf_names.update(current_leaf_names)
        individual_matrices_info.append(
            {"matrix": dist_matrix, "names": current_leaf_names}
        )

    if not all_leaf_names:  # All trees had no leaves
        return np.array([]), []

    canonical_leaf_names = sorted(list(all_leaf_names))
    name_to_index_map = {name: i for i, name in enumerate(canonical_leaf_names)}

    num_canonical_leaves = len(canonical_leaf_names)
    sum_aligned_matrices = np.zeros((num_canonical_leaves, num_canonical_leaves))
    num_valid_trees = 0

    for info in individual_matrices_info:
        current_dist_matrix = info["matrix"]
        current_leaf_names_list = info["names"]

        # Create a temporary aligned matrix for the current tree
        temp_aligned_matrix = np.zeros((num_canonical_leaves, num_canonical_leaves))

        current_name_to_temp_idx = {
            name: i for i, name in enumerate(current_leaf_names_list)
        }

        for i, name1 in enumerate(current_leaf_names_list):
            for j, name2 in enumerate(current_leaf_names_list):
                if i < j:  # Fill only upper triangle, then reflect, or fill directly
                    # Get original distance
                    dist = current_dist_matrix[i, j]

                    # Get indices in the canonical matrix
                    canonical_idx1 = name_to_index_map[name1]
                    canonical_idx2 = name_to_index_map[name2]

                    temp_aligned_matrix[canonical_idx1, canonical_idx2] = dist
                    temp_aligned_matrix[canonical_idx2, canonical_idx1] = dist

        sum_aligned_matrices += temp_aligned_matrix
        num_valid_trees += 1

    if num_valid_trees == 0:
        return np.array([]), []

    consensus_matrix = sum_aligned_matrices / num_valid_trees
    return consensus_matrix, canonical_leaf_names


def fiedler_ordering_for_tree_pair(trees: List[Node]) -> tuple:
    """
    Compute the consensus Fiedler vector ordering for a pair (or list) of trees and reorder all trees.
    Returns (ordering, fiedler_scores).
    """
    consensus, leaf_names = consensus_distance_matrix(trees)
    L = laplacian(consensus, normed=True)
    eigvals, eigvecs = eigh(L)
    fiedler_vec = eigvecs[:, 1]
    fiedler_scores = dict(zip(leaf_names, fiedler_vec))
    ordering = [name for _, name in sorted(zip(fiedler_vec, leaf_names))]
    for tree in trees:
        order_internal_nodes_by_fiedler(tree, fiedler_scores)
    return ordering, fiedler_scores
