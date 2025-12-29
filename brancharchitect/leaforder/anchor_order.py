"""Anchor-based leaf ordering for phylogenetic trees.

This module implements an anchor-based ordering algorithm that uses the lattice
algorithm to identify jumping taxa and positions them at the extremes of the
ordering while maintaining stable "anchor" taxa in their relative positions.
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional, List
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree import Node, ReorderStrategy
from brancharchitect.jumping_taxa.lattice.mapping import (
    map_solution_elements_to_minimal_frontiers,
)
from brancharchitect.jumping_taxa.lattice.solvers.lattice_solver import (
    LatticeSolver,
)
from brancharchitect.logger import jt_logger

__all__ = [
    "derive_order_for_pair",
    "blocked_order_and_apply",
]


# Per-edge caches to keep rotations and mover ranks stable across repeated calls
_rotation_cut_cache: Dict[Tuple[int, ...], Tuple[int, int, Tuple[str, ...]]] = {}
_mover_rank_cache: Dict[
    Tuple[int, ...], Dict[Tuple[int, ...], Tuple[int, int, int]]
] = {}


# ---------------------------------------------------------------------------
# Private helper functions
# ---------------------------------------------------------------------------


def _sample_pairs(
    m: Dict[Partition, Partition], k: int = 5
) -> List[Tuple[List[str], List[str]]]:
    """Sample k pairs from a mapping dictionary for debugging purposes.

    Args:
        m: Dictionary mapping source partitions to mapped partitions
        k: Maximum number of pairs to sample

    Returns:
        List of tuples containing sorted taxa lists from source and mapped partitions
    """
    pairs: List[Tuple[List[str], List[str]]] = []
    for i, (sol, mapped) in enumerate(m.items()):
        if i >= k:
            break
        pairs.append((sorted(list(sol.taxa)), sorted(list(mapped.taxa))))
    return pairs


def _rotate_list(lst: List[str], k: int) -> List[str]:
    """Rotate a list by k positions to the left.

    Args:
        lst: List to rotate
        k: Number of positions to rotate (positive = left rotation)

    Returns:
        New rotated list (original list unchanged)
    """
    if not lst:
        return lst
    k %= len(lst)
    return lst[k:] + lst[:k]


def _boundary_between_anchor_blocks(
    order: List[str], key_map: Dict[str, Tuple[int, int, int]]
) -> int:
    """Find boundary index between different anchor blocks in circular ordering.

    Searches for an adjacency i | i+1 where both taxa are anchors (band=1)
    but belong to different blocks (different anchor_pos).

    Args:
        order: Ordered list of taxa names
        key_map: Mapping from taxon to sort key tuple (band, anchor_pos, within_block_pos)

    Returns:
        Index where the cut should be made (0 if no suitable boundary found)
    """
    n = len(order)
    if n == 0:
        return 0
    for i in range(n):
        a = order[i]
        b = order[(i + 1) % n]
        band_a, anchor_pos_a, _ = key_map[a]
        band_b, anchor_pos_b, _ = key_map[b]
        if band_a == 1 and band_b == 1 and anchor_pos_a != anchor_pos_b:
            return (i + 1) % n
    # Fallback: cut at a band change
    for i in range(n):
        a = order[i]
        b = order[(i + 1) % n]
        if key_map[a][0] != key_map[b][0]:
            return (i + 1) % n
    return 0


def _boundary_largest_mover_at_zero(
    order: List[str], mover_blocks: List[Partition]
) -> int:
    """Find boundary index to place largest mover block at position zero.

    Args:
        order: Ordered list of taxa names
        mover_blocks: List of jumping taxa partitions

    Returns:
        Index of first taxon in the largest mover block (0 if no movers)
    """
    if not mover_blocks:
        return 0
    # Choose largest mover (by size; then by indices for determinism)
    largest = sorted(mover_blocks, key=lambda p: (-len(p.indices), p.indices))[0]
    block_taxa = set(largest.taxa)
    for i, t in enumerate(order):
        if t in block_taxa:
            return i
    return 0


def _cached_mover_assignments(
    edge: Partition,
    mover_blocks: List[Partition],
    mover_weight_policy: str,
) -> Dict[Tuple[int, ...], Tuple[int, int, int]]:
    """Return stable band/rank assignments for mover blocks.

    All movers go to the same side to minimize anchor displacement:
    - All movers: band 0 in source (left), band 2 in destination (right)

    This ensures anchors stay stable in the middle and movers move
    independently without crossing each other.
    """
    edge_key = (tuple(edge.indices), mover_weight_policy)
    composition = {tuple(p.indices) for p in mover_blocks}
    cached = _mover_rank_cache.get(edge_key)
    if cached and set(cached.keys()) == composition:
        return cached

    assignments: Dict[Tuple[int, ...], Tuple[int, int, int]] = {}
    jumping_count = len(mover_blocks)

    for i, jumping_partition in enumerate(mover_blocks):
        if mover_weight_policy not in ("increasing", "decreasing"):
            mover_weight_policy = "increasing"
        rank = i if mover_weight_policy == "increasing" else (jumping_count - i)

        # All movers go to the same side: left in source, right in destination
        # This minimizes anchor displacement
        src_band = 0  # left in source
        dst_band = 2  # right in destination

        assignments[tuple(jumping_partition.indices)] = (src_band, dst_band, rank)

    _mover_rank_cache[edge_key] = assignments
    return assignments


def _get_solution_mappings(
    t1: Node,
    t2: Node,
    precomputed_solution: Optional[Dict[Partition, List[Partition]]] = None,
) -> Tuple[
    Dict[Partition, Dict[Partition, Partition]],
    Dict[Partition, Dict[Partition, Partition]],
]:
    r"""Calculate per-pivot solution mappings using pivot-scoped minimal unique splits.

    For each pivot edge from compute_pivot_solutions_with_deletions, compute the unique splits
    locally under that pivot (t1_pivot_splits \ t2_pivot_splits and vice versa),
    take their minimal elements (the unique frontiers), and map the pivot's
    jumping-taxa solution partitions onto these minimal frontiers for t1 and t2.
    """
    if precomputed_solution is not None:
        solutions_by_edge = precomputed_solution
    else:
        solutions_by_edge, _ = LatticeSolver(t1, t2).solve_iteratively()

    mapped_t1: Dict[Partition, Dict[Partition, Partition]] = {}
    mapped_t2: Dict[Partition, Dict[Partition, Partition]] = {}

    for edge, solution_sets in solutions_by_edge.items():
        t1_node = t1.find_node_by_split(edge)
        t2_node = t2.find_node_by_split(edge)
        if not t1_node or not t2_node:
            continue

        # Pivot-scoped unique splits
        t1_local_splits = t1_node.to_splits()
        t2_local_splits = t2_node.to_splits()
        unique_splits_t1 = t1_local_splits - t2_local_splits
        unique_splits_t2 = t2_local_splits - t1_local_splits

        # Map ONLY this pivot's solutions using minimal unique frontiers
        per_t1, per_t2 = map_solution_elements_to_minimal_frontiers(
            {edge: solution_sets},
            unique_splits_t1=unique_splits_t1,
            unique_splits_t2=unique_splits_t2,
        )

        if edge in per_t1:
            mapped_t1[edge] = per_t1[edge]
        if edge in per_t2:
            mapped_t2[edge] = per_t2[edge]

    return mapped_t1, mapped_t2


def derive_order_for_pair(
    t1: Node,
    t2: Node,
    mappings_t1: Optional[Dict[Partition, Dict[Partition, Partition]]] = None,
    mappings_t2: Optional[Dict[Partition, Dict[Partition, Partition]]] = None,
    mover_weight_policy: str = "decreasing",
    anchor_weight_policy: str = "preserve_source",
    circular: bool = False,
    circular_boundary_policy: str = "between_anchor_blocks",
    precomputed_solution: Optional[Dict[Partition, List[Partition]]] = None,
):
    """
    Derives and applies leaf orderings for all differing edges between two trees.

    If solution mappings are not provided, they are calculated first.

    For identical trees (no mappings), still applies ordering to ensure alignment.
    """
    if mappings_t1 is None or mappings_t2 is None:
        mappings_t1, mappings_t2 = _get_solution_mappings(
            t1, t2, precomputed_solution=precomputed_solution
        )

    if not jt_logger.disabled:
        jt_logger.info("Source maps + derived jumping taxa per edge:")

    # Apply ordering for differing edges
    for edge, mapping in mappings_t1.items():
        if not jt_logger.disabled:
            try:
                dst_map = mappings_t2.get(edge, {})
                jt_logger.info(
                    f"\n[anchor_order] edge={list(edge.indices)} src_map={len(mapping)} dst_map={len(dst_map)}"
                )

                jt_logger.info(
                    f"  src pairs (solution -> mapped) sample: {_sample_pairs(mapping)}"
                )
                jt_logger.info(
                    f"  dst pairs (solution -> mapped) sample: {_sample_pairs(dst_map)}"
                )
            except Exception:
                pass
        blocked_order_and_apply(
            edge,
            mapping,
            mappings_t2.get(edge, {}),
            t1,
            t2,
            mover_weight_policy=mover_weight_policy,
            anchor_weight_policy=anchor_weight_policy,
            circular=circular,
            circular_boundary_policy=circular_boundary_policy,
        )

    # For identical trees (no mappings), still apply ordering to ensure alignment
    if not mappings_t1:
        if not jt_logger.disabled:
            jt_logger.info(
                "No differing edges found - trees may be identical. Applying root-level alignment."
            )
        # Create a root partition for the entire tree (all taxa)
        all_taxa_indices = tuple(sorted(t1.taxa_encoding.values()))
        root_partition = Partition(all_taxa_indices, t1.taxa_encoding)
        blocked_order_and_apply(
            root_partition,
            {},  # No sources
            {},  # No destinations
            t1,
            t2,
            mover_weight_policy=mover_weight_policy,
            anchor_weight_policy=anchor_weight_policy,
            circular=circular,
            circular_boundary_policy=circular_boundary_policy,
        )


def blocked_order_and_apply(
    edge: Partition,
    sources: Dict[Partition, Partition],
    destinations: Dict[Partition, Partition],
    t1: Node,
    t2: Node,
    mover_weight_policy: str = "decreasing",
    anchor_weight_policy: str = "preserve_source",
    circular: bool = False,
    circular_boundary_policy: str = "between_anchor_blocks",
):
    """
    Derive and apply a new leaf order to the subtrees defined by an edge.

    This function calculates a new ordering for the taxa in the subtrees
    (rooted at `src_node` and `dst_node`) corresponding to the given `edge`.
    The ordering is designed to group jumping-taxa partitions and place them at
    the extremes of the ordering, with stable "anchor" taxa in between.

    The `t1` and `t2` nodes are modified in-place.
    """

    src_node = t1.find_node_by_split(edge)
    dst_node = t2.find_node_by_split(edge)

    if not src_node or not dst_node:
        raise ValueError(
            f"Pivot edge not found in one or both trees: {edge}. "
            "All callers must supply an existing pivot split (shared by t1 and t2)."
        )

    # Get the current order from both trees to preserve internal structure
    src_current_order = src_node.get_current_order()
    dst_current_order = dst_node.get_current_order()

    # If the pivot subtree is already aligned and there are no jumping partitions,
    # leave it untouched to keep common subtrees stable.
    if not sources and not destinations and src_current_order == dst_current_order:
        return

    destination_index = {taxon: i for i, taxon in enumerate(dst_current_order)}
    source_index = {taxon: i for i, taxon in enumerate(src_current_order)}

    # Include leaves (trivial splits) to ensure we capture ALL common taxa
    common_splits = src_node.to_splits(with_leaves=True).intersection(
        dst_node.to_splits(with_leaves=True)
    ) - {edge}

    # Collect ALL jumping-taxa partitions using SOLUTION KEYS (mapping keys)
    # These represent the jumping partitions; exclude the pivot edge itself
    jumping_taxa_partitions_set = set(sources.keys()) | set(destinations.keys())
    jumping_taxa_partitions_set = {p for p in jumping_taxa_partitions_set if p != edge}
    # Convert to sorted list for deterministic iteration order
    # Sort by DESCENDING size so larger groups move first (they typically have smaller expand paths)
    jumping_taxa_partitions = sorted(
        jumping_taxa_partitions_set, key=lambda p: (-len(p.indices), p.indices)
    )

    # CRITICAL: Remove jumping partitions from common_splits to get only STABLE common clades
    # Jumping partitions are common clades that appear in both trees but at different positions
    # We want to treat them separately with extreme weights
    stable_common_splits = common_splits - jumping_taxa_partitions_set

    # Use maximal_elements() to get maximal stable subtrees - the largest common blocks for ordering
    stable_common_splits: PartitionSet[Partition] = (
        stable_common_splits.maximal_elements()
    )

    # Get taxa that are in jumping partitions
    jumping_taxa = {taxon for jp in jumping_taxa_partitions for taxon in jp.taxa}

    # Create a unified list of all stable blocks (stable common splits and free leaves).
    # This is a more elegant way to represent all the non-jumping parts.
    taxa_in_stable_blocks = {taxon for s in stable_common_splits for taxon in s.taxa}
    # Exclude jumping taxa from free_taxa - jumping taxa get their own extreme weights
    free_taxa = set(edge.taxa) - taxa_in_stable_blocks - jumping_taxa

    # If no anchors exist (star-like case), seed a deterministic synthetic anchor
    if not stable_common_splits and free_taxa:
        sentinel_taxon = min(
            free_taxa, key=lambda t: (source_index.get(t, float("inf")), t)
        )
        sentinel_index = t1.taxa_encoding.get(sentinel_taxon)
        if sentinel_index is not None:
            stable_common_splits = PartitionSet(
                [Partition((sentinel_index,), t1.taxa_encoding)]
            )
            free_taxa.remove(sentinel_taxon)

    # Build blocks: stable common splits preserve their current order; free taxa are singletons
    source_blocked: List[Tuple[str, ...]] = []
    for cs in stable_common_splits:
        node = t1.find_node_by_split(cs)
        if node is not None:
            source_blocked.append(tuple(node.get_current_order()))
        else:
            if not jt_logger.disabled:
                jt_logger.warning(
                    f"Warning: Could not find node for common split {cs} in tree 1"
                )
    source_blocked.extend([(taxon,) for taxon in sorted(list(free_taxa))])

    # Tuple-based sort keys per taxon to avoid large numeric weights and floats.
    # Key = (band, anchor_pos_or_rank, within_block_pos)
    # Bands: 0 = left extreme (jumping), 1 = anchors, 2 = right extreme (jumping)
    src_taxon_sort_key: Dict[str, Tuple[int, int, int]] = {}
    dst_taxon_sort_key: Dict[str, Tuple[int, int, int]] = {}

    for block in source_blocked:
        # Determine anchor position for this stable block
        if anchor_weight_policy == "preserve_source":
            src_anchor_pos = min(source_index[t] for t in block)
            dst_anchor_pos = min(destination_index[t] for t in block)
        else:
            # destination policy keeps anchors in the same order in both trees
            src_anchor_pos = min(destination_index[t] for t in block)
            dst_anchor_pos = src_anchor_pos

        # Band 1 for anchors; within-block ordering is tree-local
        ordered_src_block = sorted(block, key=lambda t: source_index[t])
        ordered_dst_block = sorted(block, key=lambda t: destination_index[t])
        for pos, taxon in enumerate(ordered_src_block):
            src_taxon_sort_key[taxon] = (1, src_anchor_pos, pos)
        for pos, taxon in enumerate(ordered_dst_block):
            dst_taxon_sort_key[taxon] = (1, dst_anchor_pos, pos)

    # Collect all anchor taxa for position-aware mover assignment
    anchor_taxa = {taxon for block in source_blocked for taxon in block}

    # Assign banded tuple keys to jumping partitions using deterministic alternation
    mover_assignments = _cached_mover_assignments(
        edge, jumping_taxa_partitions, mover_weight_policy
    )
    for i, jumping_partition in enumerate(jumping_taxa_partitions):
        src_band, dst_band, rank = mover_assignments[tuple(jumping_partition.indices)]
        # Within-block order : tree-local
        ordered_src_block = sorted(
            jumping_partition.taxa, key=lambda t: source_index[t]
        )
        ordered_dst_block = sorted(
            jumping_partition.taxa, key=lambda t: destination_index[t]
        )
        for pos, taxon in enumerate(ordered_src_block):
            # Direction-aware rank: left band wants larger rank first (more extreme left)
            second_key = -rank if src_band == 0 else rank
            src_taxon_sort_key[taxon] = (src_band, second_key, pos)
        for pos, taxon in enumerate(ordered_dst_block):
            second_key = -rank if dst_band == 0 else rank
            dst_taxon_sort_key[taxon] = (dst_band, second_key, pos)

    # Build final taxa lists and sort using tuple keys
    all_taxa_in_edge = list(edge.taxa)
    sorted_src_taxa = sorted(all_taxa_in_edge, key=lambda t: src_taxon_sort_key[t])
    sorted_dest_taxa = sorted(all_taxa_in_edge, key=lambda t: dst_taxon_sort_key[t])

    # Optional circular rotation of the final permutations for circular rendering
    if circular:
        edge_key = tuple(edge.indices)
        if circular_boundary_policy == "largest_mover_at_zero":
            src_cut_candidate = _boundary_largest_mover_at_zero(
                sorted_src_taxa, jumping_taxa_partitions
            )
            dst_cut_candidate = _boundary_largest_mover_at_zero(
                sorted_dest_taxa, jumping_taxa_partitions
            )
        else:
            src_cut_candidate = _boundary_between_anchor_blocks(
                sorted_src_taxa, src_taxon_sort_key
            )
            dst_cut_candidate = _boundary_between_anchor_blocks(
                sorted_dest_taxa, dst_taxon_sort_key
            )

        cached_cuts = _rotation_cut_cache.get(edge_key)
        if cached_cuts and cached_cuts[2] == tuple(sorted_src_taxa):
            cached_src_cut, cached_dst_cut, _ = cached_cuts
            if src_cut_candidate == 0 and cached_src_cut:
                src_cut_candidate = cached_src_cut
            if dst_cut_candidate == 0 and cached_dst_cut:
                dst_cut_candidate = cached_dst_cut

        src_cut = src_cut_candidate
        dst_cut = dst_cut_candidate
        _rotation_cut_cache[edge_key] = (
            src_cut,
            dst_cut,
            tuple(sorted_src_taxa),
        )

        sorted_src_taxa = _rotate_list(sorted_src_taxa, src_cut)
        sorted_dest_taxa = _rotate_list(sorted_dest_taxa, dst_cut)

    src_node.reorder_taxa(sorted_src_taxa, ReorderStrategy.MINIMUM)
    dst_node.reorder_taxa(sorted_dest_taxa, ReorderStrategy.MINIMUM)
