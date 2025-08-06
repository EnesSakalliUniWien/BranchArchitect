# =============================================================================
# ENHANCED GLOBAL OPTIMIZATION (PHYLO-IO INSPIRED)
# =============================================================================

from typing import Dict, Tuple, List
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree import Node
from brancharchitect.elements.partition import Partition
from .core_rooting import reroot_at_node


def _filter_and_score_split_candidates(
    split_a: Partition,
    splits_b: PartitionSet[Partition],
    similarity_matrix: Dict[Tuple[Partition, Partition], float],
    threshold: float = 0.1,
) -> List[Tuple[Partition, float]]:
    """
    Filter and score potential split candidates based on similarity threshold.

    Args:
        split_a: The split to find matches for
        splits_b: Set of potential matching splits
        similarity_matrix: Precomputed similarity scores
        threshold: Minimum similarity threshold

    Returns:
        List of (split, score) tuples above threshold, sorted by score
    """
    candidates: List[Tuple[Partition, float]] = []

    for split_b in splits_b:
        similarity: float = similarity_matrix.get((split_a, split_b), 0.0)
        if similarity >= threshold:
            candidates.append((split_b, similarity))

    # Sort by similarity score (descending)
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates


def build_global_correspondence_map(
    splits_a: PartitionSet[Partition],
    splits_b: PartitionSet[Partition],
    similarity_matrix: Dict[Tuple[Partition, Partition], float],
) -> Dict[Partition, Partition]:
    """
    Build a global correspondence map between splits from two trees.

    Args:
        splits_a: Splits from first tree
        splits_b: Splits from second tree
        similarity_matrix: Precomputed similarity matrix

    Returns:
        Dictionary mapping splits from tree A to best matches in tree B
    """
    correspondence_map: Dict[Partition, Partition] = {}
    used_splits_b: set[Partition] = set()

    # Sort splits_a by some criterion (e.g., size) for consistent processing
    sorted_splits_a: List[Partition] = sorted(
        splits_a, key=lambda x: len(getattr(x, "indices", []))
    )

    for split_a in sorted_splits_a:
        # Get candidates above threshold
        candidates = _filter_and_score_split_candidates(
            split_a, splits_b, similarity_matrix
        )

        # Find best unused candidate
        for split_b, _score in candidates:
            if split_b not in used_splits_b:
                correspondence_map[split_a] = split_b
                used_splits_b.add(split_b)
                break

    return correspondence_map


def _calculate_split_weight(split: Partition) -> float:
    """
    Calculate a weight for a split based on its characteristics.

    Args:
        split: The split to calculate weight for

    Returns:
        Weight value (higher = more important)
    """
    # Base weight on split size (avoid trivial splits)
    if hasattr(split, "indices"):
        size: int = len(split.indices)
        total_size = getattr(split, "total_taxa", size * 2)  # Estimate if not available

        # Weight splits that are more balanced
        if total_size > 0:
            balance = min(size, total_size - size) / (total_size / 2)
            return balance

    return 1.0  # Default weight


def _get_best_similarity_for_split(
    split: Partition,
    other_splits: PartitionSet[Partition],
    similarity_matrix: Dict[Tuple[Partition, Partition], float],
) -> float:
    """
    Get the best similarity score for a split against a set of other splits.

    Args:
        split: The split to find best match for
        other_splits: Set of splits to compare against
        similarity_matrix: Precomputed similarity scores

    Returns:
        Best similarity score found
    """
    best_similarity = 0.0

    for other_split in other_splits:
        similarity: float = similarity_matrix.get((split, other_split), 0.0)
        best_similarity: float = max(best_similarity, similarity)

    return best_similarity


def _compute_global_similarity_score_splits(
    splits_a: PartitionSet[Partition],
    splits_b: PartitionSet[Partition],
    similarity_matrix: Dict[Tuple[Partition, Partition], float],
) -> float:
    """
    Compute a global similarity score between two sets of splits.

    Args:
        splits_a: First set of splits
        splits_b: Second set of splits
        similarity_matrix: Precomputed similarity matrix

    Returns:
        Global similarity score
    """
    if not splits_a or not splits_b:
        return 0.0

    total_weighted_similarity = 0.0
    total_weight = 0.0

    for split_a in splits_a:
        weight: float = _calculate_split_weight(split_a)
        best_similarity: float = _get_best_similarity_for_split(
            split_a, splits_b, similarity_matrix
        )

        total_weighted_similarity += weight * best_similarity
        total_weight += weight

    return total_weighted_similarity / total_weight if total_weight > 0 else 0.0


def get_split_to_node_mapping_for_root_search(tree: Node) -> Dict[Partition, Node]:
    """
    Create a mapping from splits to their corresponding nodes for root search.

    Args:
        tree: Tree to create mapping for

    Returns:
        Dictionary mapping splits to nodes
    """
    split_to_node: Dict[Partition, Node] = {}

    for node in tree.traverse():
        if node.split_indices and len(node.split_indices.indices) > 0:
            split_to_node[node.split_indices] = node

    return split_to_node


def evaluate_split_as_root_candidate(
    node: Node,
    reference_splits: PartitionSet[Partition],
    similarity_matrix: Dict[Tuple[Partition, Partition], float],
) -> float:
    """
    Evaluate how good a split would be as a root candidate.

    Args:
        split: The split to evaluate
        node: The node corresponding to this split
        reference_splits: Reference splits to compare against
        similarity_matrix: Precomputed similarity scores

    Returns:
        Score indicating quality as root candidate
    """
    # Get the tree rooted at this node
    temp_root: Node = reroot_at_node(node)

    # Extract splits from temporarily rerooted tree
    temp_splits: PartitionSet[Partition] = PartitionSet()
    for temp_node in temp_root.traverse():
        if temp_node.split_indices and len(temp_node.split_indices.indices) > 0:
            temp_splits.add(temp_node.split_indices)

    # Compute global similarity
    score: float = _compute_global_similarity_score_splits(
        temp_splits, reference_splits, similarity_matrix
    )

    return score
