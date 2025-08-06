from typing import Optional, Tuple
from brancharchitect.tree import Node
from brancharchitect.elements.partition import Partition
from .core_rooting import reroot_at_node

# =============================================================================
# JACCARD SIMILARITY-BASED MATCHING
# =============================================================================


def _get_target_indices_and_bitmask(
    target_partition: Partition,
) -> Tuple[Optional[int], Optional[set[int]]]:
    """
    Extract bitmask and indices from target partition for efficient comparison.

    Args:
        target_partition: The partition to extract data from

    Returns:
        Tuple of (bitmask_popcount, indices_set) where only one is not None
    """
    if not target_partition:
        return None, None

    # Partition objects always have both bitmask and indices attributes
    popcount: int = bin(target_partition.bitmask).count("1")
    return popcount, None


def _calculate_jaccard_similarity(
    node_partition: Partition,
    target_partition: Partition,
) -> float:
    """
    Calculate Jaccard similarity between node partition and target partition.

    Args:
        node_partition: The node's partition
        target_popcount: Precomputed popcount of target bitmask (unused in optimized version)
        target_indices: Set of target indices (unused in optimized version)
        target_partition: Original target partition for bitmask operations

    Returns:
        Jaccard similarity score (0.0 to 1.0)
    """
    # Use direct bitmask operations for maximum performance
    intersection: int = bin(node_partition.bitmask & target_partition.bitmask).count(
        "1"
    )
    union: int = bin(node_partition.bitmask | target_partition.bitmask).count("1")
    return intersection / union if union > 0 else 0.0


def find_best_matching_node_jaccard(
    target_partition: Partition, root: Node
) -> Optional[Node]:
    """
    Find the node with the highest Jaccard similarity to the target partition.

    Args:
        target_partition: The partition to match against
        root: Root of the tree to search in

    Returns:
        Node with highest Jaccard similarity, or None if no valid matches
    """
    if not target_partition:
        return None

    # Extract target data once for efficiency
    target_popcount, target_indices = _get_target_indices_and_bitmask(target_partition)

    if target_popcount is None and target_indices is None:
        return None

    best_node = None
    best_similarity = 0.0

    for node in root.traverse():
        # Check if node has valid split_indices (non-empty Partition)
        if node.split_indices and len(node.split_indices.indices) > 0:
            similarity: float = _calculate_jaccard_similarity(
                node.split_indices, target_partition
            )

            if similarity > best_similarity:
                best_similarity: float = similarity
                best_node: Node = node

                # Early termination for perfect match
                if similarity >= 1.0:
                    break

    return best_node


def reroot_by_jaccard_similarity(tree1: Node, tree2: Node) -> Node:
    """
    Reroot tree1 to maximize Jaccard similarity with tree2's root partition.

    Args:
        tree1: Tree to reroot
        tree2: Reference tree

    Returns:
        Rerooted tree1
    """
    if (
        not hasattr(tree2, "split_indices")
        or not tree2.split_indices
        or len(tree2.split_indices.indices) == 0
    ):
        return tree1

    best_node: Node | None = find_best_matching_node_jaccard(tree2.split_indices, tree1)
    if best_node:
        return reroot_at_node(best_node)
    return tree1
