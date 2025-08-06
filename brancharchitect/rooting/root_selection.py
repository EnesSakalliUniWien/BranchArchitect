from typing import Dict, Tuple, List
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree import Node
from brancharchitect.elements.partition import Partition
from .global_optimization import (
    get_split_to_node_mapping_for_root_search,
    evaluate_split_as_root_candidate,
)


def _add_fallback_candidates(
    candidates: List[Tuple[Node, float]],
    tree: Node,
    max_candidates: int = 10,
) -> List[Tuple[Node, float]]:
    """
    Add fallback candidates if not enough high-quality candidates found.

    Args:
        candidates: Current list of candidates
        tree: Tree to search for additional candidates
        max_candidates: Maximum number of candidates to return

    Returns:
        Extended list of candidates
    """
    if len(candidates) >= max_candidates:
        return candidates[:max_candidates]

    # Add internal nodes as fallback candidates
    fallback_nodes: List[Tuple[Node, float]] = []
    for node in tree.traverse():
        if node.children and not any(node == cand for cand, _ in candidates):
            fallback_nodes.append((node, 0.0))  # Zero score for fallback

    # Add fallback candidates up to max limit
    remaining_slots: int = max_candidates - len(candidates)
    candidates.extend(fallback_nodes[:remaining_slots])

    return candidates


def find_optimal_root_candidates(
    tree: Node,
    reference_splits: PartitionSet[Partition],
    similarity_matrix: Dict[Tuple[Partition, Partition], float],
    max_candidates: int = 5,
) -> List[Tuple[Node, float]]:
    """
    Find optimal root candidates based on global split similarity.

    Args:
        tree: Tree to find root candidates for
        reference_splits: Reference splits to optimize against
        similarity_matrix: Precomputed similarity matrix
        max_candidates: Maximum number of candidates to return

    Returns:
        List of (node, score) tuples, sorted by score (descending)
    """
    # Get mapping from splits to nodes
    split_to_node: Dict[Partition, Node] = get_split_to_node_mapping_for_root_search(
        tree
    )

    candidates: List[Tuple[Node, float]] = []

    # Evaluate each split as a potential root
    for _split, node in split_to_node.items():
        score = evaluate_split_as_root_candidate(
            node, reference_splits, similarity_matrix
        )
        candidates.append((node, score))

    # Sort by score (descending)
    candidates.sort(key=lambda x: x[1], reverse=True)

    # Add fallback candidates if needed
    candidates = _add_fallback_candidates(candidates, tree, max_candidates)

    return candidates[:max_candidates]
