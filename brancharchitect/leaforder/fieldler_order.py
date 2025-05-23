import logging
from typing import List, Tuple, Dict
from brancharchitect.tree import Node
from brancharchitect.partition_set import PartitionSet
from brancharchitect.leaforder.old.tree_order_optimisation_local import (
    build_orientation_map,
    reorder_tree_if_full_common,
)
from brancharchitect.leaforder.rotation_functions import (
    get_unique_splits,
    get_s_edge_splits,
    optimize_splits,
)
import numpy as np
from scipy.sparse.csgraph import laplacian
from scipy.linalg import eigh


# --- Spectral (Fiedler vector) ordering utilities ---

# Memoization cache for leaf distance matrices
_leaf_distance_matrix_cache = {}


def distance_between_leaves(tree: Node, leaf1: Node, leaf2: Node) -> float:
    """
    Compute the path length between two leaves in the tree.
    This implementation assumes unique leaf names and that each node has a 'length' attribute.
    """

    # Find paths from root to each leaf
    def path_to_root(leaf):
        path = []
        node = leaf
        while node is not None:
            path.append(node)
            node = node.parent
        return path[::-1]  # from root to leaf

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
    leaves = tree.get_leaves()
    n = len(leaves)
    dist_matrix = np.zeros((n, n))
    for i, leaf1 in enumerate(leaves):
        for j, leaf2 in enumerate(leaves):
            if i < j:
                dist = distance_between_leaves(tree, leaf1, leaf2)
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
    Returns (consensus, leaf_names).
    """
    matrices = []
    for tree in trees:
        dist_matrix, _ = compute_leaf_distance_matrix(tree)
        matrices.append(dist_matrix)
    consensus = np.mean(matrices, axis=0)
    leaf_names = [leaf.name for leaf in trees[0].get_leaves()]
    return consensus, leaf_names


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
