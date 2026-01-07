"""Anchor-based leaf ordering for phylogenetic trees.

Implements a 3-band ordering strategy:
1. Band 0 (Left): Jumping taxa moving left.
2. Band 1 (Center): Stable "anchor" subtrees common to both trees.
3. Band 2 (Right): Jumping taxa moving right.

This separation minimizes visual crossing ("hairball effect") during animation.
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional, List
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree import Node, ReorderStrategy
from brancharchitect.jumping_taxa.lattice.mapping import (
    map_solution_elements_via_parent,
)
from brancharchitect.jumping_taxa.lattice.solvers.lattice_solver import (
    LatticeSolver,
)
from brancharchitect.jumping_taxa.lattice.ordering.edge_depth_ordering import (
    topological_sort_edges,
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


def _rotate_list(lst: List[str], k: int) -> List[str]:
    """Rotate a list by k positions to the left.

    Args:
        lst: List to rotate
        k: Number of positions to rotate (positive = left rotation)

    Returns:
        New rotated list (original list unchanged)
    """
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
    for i in range(n):
        a = order[i]
        b = order[(i + 1) % n]
        band_a, anchor_pos_a, _ = key_map.get(a, (1, 0, 0))
        band_b, anchor_pos_b, _ = key_map.get(b, (1, 0, 0))
        if band_a == 1 and band_b == 1 and anchor_pos_a != anchor_pos_b:
            return (i + 1) % n
    # Fallback: cut at a band change
    for i in range(n):
        a = order[i]
        b = order[(i + 1) % n]
        if key_map.get(a, (1, 0, 0))[0] != key_map.get(b, (1, 0, 0))[0]:
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
    source_index: Optional[Dict[str, int]] = None,
    destination_index: Optional[Dict[str, int]] = None,
) -> Dict[Tuple[int, ...], Tuple[int, int, int]]:
    """Return stable band/rank assignments for mover blocks.

    Independent movers are assigned to alternating bands so they can move
    independently in circular layouts:
    - First mover: band 0 in source (left), band 2 in destination (right)
    - Second mover: band 2 in source (right), band 0 in destination (left)
    - And so on...

    This ensures independent movers don't appear to move together visually.
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

        # Alternate bands for independent movers so they move independently
        # Even-indexed movers: left in source, right in destination
        # Odd-indexed movers: right in source, left in destination
        if i % 2 == 0:
            src_band = 0  # left in source
            dst_band = 2  # right in destination
        else:
            src_band = 2  # right in source
            dst_band = 0  # left in destination

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
    r"""Calculate per-pivot solution mappings using parent relationships.

    For each pivot edge, map the pivot's jumping-taxa solution partitions
    to their parent nodes in t1 and t2, directly showing where each subtree
    is attached in each tree.
    """
    if precomputed_solution is not None:
        solutions_by_edge = precomputed_solution
    else:
        solutions_by_edge, _ = LatticeSolver(t1, t2).solve_iteratively()

    # Use the simpler parent-based mapping
    mapped_t1, mapped_t2 = map_solution_elements_via_parent(solutions_by_edge, t1, t2)

    return mapped_t1, mapped_t2


def _get_stable_and_moving_components(
    edge: Partition,
    src_node: Node,
    dst_node: Node,
    sources: Dict[Partition, Partition],
    destinations: Dict[Partition, Partition],
    source_index: Dict[str, int],
    t1: Node,
    common_splits: Optional[PartitionSet[Partition]] = None,
) -> Tuple[List[Tuple[str, ...]], List[Partition]]:
    """Identify stable anchor blocks and jumping mover partitions."""
    # Include leaves (trivial splits) to ensure we capture ALL common taxa
    if common_splits is not None:
        # Optimization: Reuse precomputed common splits
        # Stable splits must be in both common_splits AND the source subtree
        stable_common_candidates = common_splits.intersection(
            src_node.to_splits(with_leaves=True)
        )
        common_splits_in_subtree = stable_common_candidates - {edge}
    else:
        # Fallback: Compute intersection locally
        common_splits_in_subtree = src_node.to_splits(with_leaves=True).intersection(
            dst_node.to_splits(with_leaves=True)
        ) - {edge}

    # Collect ALL jumping-taxa partitions using SOLUTION KEYS (mapping keys)
    # These represent the jumping partitions; exclude the pivot edge itself
    jumping_taxa_partitions_set = set(sources.keys()) | set(destinations.keys())
    jumping_taxa_partitions_set = {p for p in jumping_taxa_partitions_set if p != edge}
    # Convert to sorted list for deterministic iteration order
    # Sort by DESCENDING size so larger groups move first
    jumping_taxa_partitions = sorted(
        jumping_taxa_partitions_set, key=lambda p: (-len(p.indices), p.indices)
    )

    # CRITICAL: Separate stable anchors from jumping movers.
    stable_common_splits = common_splits_in_subtree - jumping_taxa_partitions_set

    # Use maximal_elements() to get maximal stable subtrees
    stable_common_splits: PartitionSet[Partition] = (
        stable_common_splits.maximal_elements()
    )

    # Build blocks: stable common splits preserve their current order
    source_blocked: List[Tuple[str, ...]] = []
    for cs in stable_common_splits:
        node = t1.find_node_by_split(cs)
        source_blocked.append(tuple(node.get_current_order()))

    return source_blocked, jumping_taxa_partitions


def _assign_anchor_keys(
    source_blocked: List[Tuple[str, ...]],
    source_index: Dict[str, int],
    destination_index: Dict[str, int],
    anchor_weight_policy: str,
    src_taxon_sort_key: Dict[str, Tuple[int, int, int]],
    dst_taxon_sort_key: Dict[str, Tuple[int, int, int]],
) -> None:
    """Assign sort keys for stable anchor blocks (Band 1)."""
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


def _assign_mover_keys(
    edge: Partition,
    jumping_taxa_partitions: List[Partition],
    mover_weight_policy: str,
    source_index: Dict[str, int],
    destination_index: Dict[str, int],
    src_taxon_sort_key: Dict[str, Tuple[int, int, int]],
    dst_taxon_sort_key: Dict[str, Tuple[int, int, int]],
) -> None:
    """Assign sort keys for jumping mover partitions (Band 0/2)."""
    mover_assignments = _cached_mover_assignments(
        edge, jumping_taxa_partitions, mover_weight_policy
    )
    for jumping_partition in jumping_taxa_partitions:
        src_band, dst_band, rank = mover_assignments[tuple(jumping_partition.indices)]

        # Within-block order : tree-local
        ordered_src_block = sorted(
            jumping_partition.taxa, key=lambda t: source_index[t]
        )
        ordered_dst_block = sorted(
            jumping_partition.taxa, key=lambda t: destination_index[t]
        )
        for pos, taxon in enumerate(ordered_src_block):
            # Direction-aware rank:
            # Band 0 (Left): Larger rank -> More negative -> More left (extreme)
            # Band 2 (Right): Larger rank -> More positive -> More right (extreme)
            second_key = -rank if src_band == 0 else rank
            src_taxon_sort_key[taxon] = (src_band, second_key, pos)
        for pos, taxon in enumerate(ordered_dst_block):
            second_key = -rank if dst_band == 0 else rank
            dst_taxon_sort_key[taxon] = (dst_band, second_key, pos)


def _handle_circular_rotation(
    edge: Partition,
    sorted_src_taxa: List[str],
    sorted_dest_taxa: List[str],
    jumping_taxa_partitions: List[Partition],
    src_taxon_sort_key: Dict[str, Tuple[int, int, int]],
    dst_taxon_sort_key: Dict[str, Tuple[int, int, int]],
    circular_boundary_policy: str,
) -> Tuple[List[str], List[str], int]:
    """Apply circular rotation to the final permutations if needed."""
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

    return sorted_src_taxa, sorted_dest_taxa, src_cut


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
    common_splits: Optional[PartitionSet[Partition]] = None,
):
    """
    Derives and applies leaf orderings for all differing edges between two trees.

    1. Calculates solution mappings if not provided.
    2. Applies `blocked_order_and_apply` to each differing edge.
    3. Applies `blocked_order_and_apply` to the root to handle global structure.
    """
    if mappings_t1 is None or mappings_t2 is None:
        mappings_t1, mappings_t2 = _get_solution_mappings(
            t1, t2, precomputed_solution=precomputed_solution
        )

    if not jt_logger.disabled:
        jt_logger.info("Source maps + derived jumping taxa per edge:")

    # Sort pivot edges topologically (subsets before supersets) for correct processing order
    ordered_edges = topological_sort_edges(list(mappings_t1.keys()), t1)

    # Apply ordering for differing edges in topological order
    for edge in ordered_edges:
        mapping = mappings_t1[edge]
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
            common_splits=common_splits,
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
            common_splits=common_splits,
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
    circular_boundary_policy: str = "largest_mover_at_zero",
    common_splits: Optional[PartitionSet[Partition]] = None,
):
    """
    Derive and apply a 3-band leaf order to the subtrees defined by an edge.

    1. Identifies stable common subtrees ("Anchors").
    2. Identifies jumping taxa ("Movers").
    3. Assigns Anchors to the center (Band 1).
    4. Assigns Movers to the extremes (Band 0 or 2).
    5. Reorders `t1` and `t2` in-place to match this visual structure.
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

    destination_index = {taxon: i for i, taxon in enumerate(dst_current_order)}
    source_index = {taxon: i for i, taxon in enumerate(src_current_order)}

    # Include leaves (trivial splits) to ensure we capture ALL common taxa
    source_blocked, jumping_taxa_partitions = _get_stable_and_moving_components(
        edge,
        src_node,
        dst_node,
        sources,
        destinations,
        source_index,
        t1,
        common_splits=common_splits,
    )

    # Tuple-based sort keys per taxon.
    # Key = (band, anchor_pos_or_rank, within_block_pos)
    # Bands: 0 = Left Mover, 1 = Anchor, 2 = Right Mover
    src_taxon_sort_key: Dict[str, Tuple[int, int, int]] = {}
    dst_taxon_sort_key: Dict[str, Tuple[int, int, int]] = {}

    _assign_anchor_keys(
        source_blocked,
        source_index,
        destination_index,
        anchor_weight_policy,
        src_taxon_sort_key,
        dst_taxon_sort_key,
    )

    # Assign banded tuple keys to jumping partitions using deterministic alternation
    _assign_mover_keys(
        edge,
        jumping_taxa_partitions,
        mover_weight_policy,
        source_index,
        destination_index,
        src_taxon_sort_key,
        dst_taxon_sort_key,
    )

    # Build final taxa lists and sort using tuple keys
    # Fallback (1, 0, 0) assigns unhandled taxa to band 1 (anchors) - this is intentional
    # as some taxa may not be covered by explicit anchor blocks or mover partitions
    all_taxa_in_edge = list(edge.taxa)
    sorted_src_taxa = sorted(
        all_taxa_in_edge, key=lambda t: src_taxon_sort_key.get(t, (1, 0, 0))
    )
    sorted_dest_taxa = sorted(
        all_taxa_in_edge, key=lambda t: dst_taxon_sort_key.get(t, (1, 0, 0))
    )

    # Optional circular rotation of the final permutations for circular rendering
    src_cut = 0
    if circular:
        sorted_src_taxa, sorted_dest_taxa, src_cut = _handle_circular_rotation(
            edge,
            sorted_src_taxa,
            sorted_dest_taxa,
            jumping_taxa_partitions,
            src_taxon_sort_key,
            dst_taxon_sort_key,
            circular_boundary_policy,
        )

    if not jt_logger.disabled:
        jt_logger.info(
            f"DEBUG: sorted_src_taxa sample: {sorted_src_taxa[:5]} ... {sorted_src_taxa[-5:]}"
        )
        jt_logger.info(f"DEBUG: src_cut={src_cut if circular else 'N/A'}")

    src_node.reorder_taxa(sorted_src_taxa, ReorderStrategy.MINIMUM)
    dst_node.reorder_taxa(sorted_dest_taxa, ReorderStrategy.MINIMUM)
