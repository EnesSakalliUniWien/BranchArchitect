"""Transient edge mapping utilities for lattice algorithm."""

from typing import List, Optional

from brancharchitect.tree import Node
from brancharchitect.elements.partition import Partition


def map_transient_sedges_to_original(
    s_edges_from_iteration: List[Partition],
    original_t1: Node,
    original_t2: Node,
) -> List[Optional[Partition]]:
    """
    Map s-edges from a pruned tree back to original common splits using a strict subset rule.

    An s-edge is mapped to an original common split only if its taxa are a direct
    subset of the original split's taxa. Among candidates, the smallest original
    split is chosen to ensure the tightest, most specific match.

    Args:
        s_edges_from_iteration: S-edge Partitions from one iteration.
        original_t1: The initial, unmodified first tree.
        original_t2: The initial, unmodified second tree.

    Returns:
        List of best-matching original partitions for each s-edge, or None if no
        valid mapping is found.
    """
    from brancharchitect.jumping_taxa.debug import jt_logger
    import logging

    logger = logging.getLogger(__name__)

    # Compute common splits once and impose a deterministic order for stable tie-breaking
    splits_t1 = original_t1.to_splits()
    splits_t2 = original_t2.to_splits()
    common = splits_t1 & splits_t2

    jt_logger.info(f"[MAPPING] Original tree 1 has {len(splits_t1)} splits")
    jt_logger.info(f"[MAPPING] Original tree 2 has {len(splits_t2)} splits")
    jt_logger.info(f"[MAPPING] Common splits: {len(common)}")
    logger.info(f"[MAPPING] Original tree 1 has {len(splits_t1)} splits")
    logger.info(f"[MAPPING] Original tree 2 has {len(splits_t2)} splits")
    logger.info(f"[MAPPING] Common splits: {len(common)}")

    def _popcount(x: int) -> int:
        try:
            return x.bit_count()  # Python 3.10+
        except AttributeError:
            return bin(x).count("1")

    original_common_splits = sorted(
        common, key=lambda p: (_popcount(p.bitmask), p.bitmask)
    )

    if not original_common_splits:
        return [None] * len(s_edges_from_iteration)

    # Pre-compute masks and sizes for originals
    originals_info = [
        (p, p.bitmask, _popcount(p.bitmask)) for p in original_common_splits
    ]

    mapped_partitions: List[Optional[Partition]] = []

    for s_edge in s_edges_from_iteration:
        s_mask = s_edge.bitmask
        s_size = _popcount(s_mask)

        # Skip degenerate s-edges with no taxa
        if s_size == 0:
            mapped_partitions.append(None)
            continue

        # Find all original splits that contain the s_edge as a subset
        candidate_matches: List[tuple[Partition, int]] = []
        for original_p, o_mask, o_size in originals_info:
            # Check for subset: (s_mask & o_mask) == s_mask
            if (s_mask & o_mask) == s_mask:
                candidate_matches.append((original_p, o_size))

        # If candidates were found, choose the best one (the smallest)
        if candidate_matches:
            # Sort candidates by size (smallest first) and pick the first one
            candidate_matches.sort(key=lambda x: x[1])
            best_match = candidate_matches[0][0]

            # DEBUG: Verify the best match exists in BOTH original trees
            in_t1 = original_t1.find_node_by_split(best_match) is not None
            in_t2 = original_t2.find_node_by_split(best_match) is not None

            jt_logger.info(
                f"[MAPPING] Mapped s-edge {s_edge.bipartition()} -> {best_match.bipartition()}"
            )
            jt_logger.info(f"[MAPPING]   Exists in original t1: {in_t1}")
            jt_logger.info(f"[MAPPING]   Exists in original t2: {in_t2}")
            logger.info(
                f"[MAPPING] Mapped s-edge {s_edge.bipartition()} -> {best_match.bipartition()}"
            )
            logger.info(f"[MAPPING]   Exists in original t1: {in_t1}")
            logger.info(f"[MAPPING]   Exists in original t2: {in_t2}")

            if not (in_t1 and in_t2):
                jt_logger.warning(
                    "[MAPPING]   ⚠️  BUG: Mapped to split that is NOT in common splits!"
                )
                jt_logger.warning(
                    "[MAPPING]       This should be impossible since we only search common splits"
                )
                logger.error(
                    "[MAPPING]   ⚠️  BUG: Mapped to split that is NOT in common splits!"
                )
                logger.error(
                    "[MAPPING]       This should be impossible since we only search common splits"
                )

            mapped_partitions.append(best_match)
        else:
            # If no original split contains the s_edge, there is no valid mapping
            jt_logger.info(
                f"[MAPPING] No mapping found for s-edge {s_edge.bipartition()}"
            )
            mapped_partitions.append(None)

    return mapped_partitions
