"""Anchor-based leaf ordering for phylogenetic trees.

Implements a 3-band ordering strategy:
1. Band 0 (Left): Jumping taxa moving left.
2. Band 1 (Center): Stable "anchor" subtrees common to both trees.
3. Band 2 (Right): Jumping taxa moving right.

This separation minimizes visual crossing ("hairball effect") during animation.
"""

from __future__ import annotations
import logging
import time
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
logger = logging.getLogger(__name__)
_rotation_cut_cache: Dict[Tuple[int, ...], Tuple[int, int, Tuple[str, ...]]] = {}
_mover_rank_cache: Dict[
    Tuple[Tuple[int, ...], str], Dict[Tuple[int, ...], Tuple[int, int, int]]
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


def _stable_anchor_splits_in_subtree(
    edge: Partition,
    src_node: Node,
    dst_node: Node,
    common_splits: Optional[PartitionSet[Partition]] = None,
) -> PartitionSet[Partition]:
    """Return stable structural anchors available under edge.

    When common_splits is precomputed by the optimizer, it represents shared
    structural splits for propagation. Do not add every matching leaf here:
    leaves are useful as a local fallback, but as precomputed root anchors they
    turn a pure visual rotation into a forced destination-order rewrite.
    """
    if common_splits is None:
        stable_anchor_splits = src_node.to_splits(with_leaves=True).intersection(
            dst_node.to_splits(with_leaves=True)
        )
    else:
        stable_anchor_splits = common_splits.intersection(
            src_node.to_splits(with_leaves=True)
        )

    return stable_anchor_splits - {edge}


def _get_stable_anchor_blocks_and_movers(
    edge: Partition,
    src_node: Node,
    dst_node: Node,
    solution_to_source: Dict[Partition, Partition],
    solution_to_destination: Dict[Partition, Partition],
    t1: Node,
    common_splits: Optional[PartitionSet[Partition]] = None,
    precomputed_anchor_nodes: Optional[List[Node]] = None,
    precomputed_mover_partitions: Optional[List[Partition]] = None,
) -> Tuple[List[Tuple[str, ...]], List[Partition]]:
    """Identify stable anchor blocks and mover partitions for this edge."""
    if precomputed_anchor_nodes is not None:
        stable_anchor_nodes = precomputed_anchor_nodes
        mover_partitions = precomputed_mover_partitions or []
    else:
        stable_anchor_nodes, mover_partitions = _get_stable_anchor_nodes_and_movers(
            edge,
            src_node,
            dst_node,
            solution_to_source,
            solution_to_destination,
            t1,
            common_splits=common_splits,
        )

    return _anchor_blocks_from_nodes(stable_anchor_nodes, src_node), mover_partitions


def _get_stable_anchor_nodes_and_movers(
    edge: Partition,
    src_node: Node,
    dst_node: Node,
    solution_to_source: Dict[Partition, Partition],
    solution_to_destination: Dict[Partition, Partition],
    t1: Node,
    common_splits: Optional[PartitionSet[Partition]] = None,
) -> Tuple[List[Node], List[Partition]]:
    """Identify stable anchor nodes and mover partitions for this edge."""
    stable_anchor_splits = _stable_anchor_splits_in_subtree(
        edge,
        src_node,
        dst_node,
        common_splits=common_splits,
    )

    # Collect ALL jumping-taxa partitions using SOLUTION KEYS (mapping keys)
    # These represent the jumping partitions; exclude the pivot edge itself
    moving_solution_set = set(solution_to_source.keys()) | set(
        solution_to_destination.keys()
    )
    moving_solution_set = {p for p in moving_solution_set if p != edge}
    # Convert to sorted list for deterministic iteration order
    # Sort by DESCENDING size so larger groups move first
    mover_partitions = sorted(
        moving_solution_set, key=lambda p: (-len(p.indices), p.indices)
    )

    # CRITICAL: Separate stable anchors from jumping movers.
    stable_common_splits = stable_anchor_splits - moving_solution_set

    # Use maximal_elements() to get maximal stable subtrees
    stable_common_splits = stable_common_splits.maximal_elements()

    stable_anchor_nodes: List[Node] = []
    for cs in stable_common_splits:
        node = t1.find_node_by_split(cs)
        if node:
            stable_anchor_nodes.append(node)

    return stable_anchor_nodes, mover_partitions


def _anchor_blocks_from_nodes(
    stable_anchor_nodes: List[Node],
    src_node: Node,
) -> List[Tuple[str, ...]]:
    """Build currently ordered anchor blocks from precomputed stable nodes."""
    source_position = {
        taxon: index for index, taxon in enumerate(src_node.get_current_order())
    }

    def anchor_block_key(node: Node) -> Tuple[int, int, Tuple[int, ...]]:
        partition = node.split_indices
        positions = [
            source_position[partition.reverse_encoding[idx]]
            for idx in partition.indices
            if partition.reverse_encoding[idx] in source_position
        ]
        first_position = min(positions) if positions else len(source_position)
        return (first_position, len(partition.indices), partition.indices)

    # Build blocks: stable common splits preserve their current order.
    stable_anchor_blocks: List[Tuple[str, ...]] = []
    for node in sorted(stable_anchor_nodes, key=anchor_block_key):
        stable_anchor_blocks.append(tuple(node.get_current_order()))

    return stable_anchor_blocks


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
    mover_partitions: List[Partition],
    mover_weight_policy: str,
    source_index: Dict[str, int],
    destination_index: Dict[str, int],
    src_taxon_sort_key: Dict[str, Tuple[int, int, int]],
    dst_taxon_sort_key: Dict[str, Tuple[int, int, int]],
) -> None:
    """Assign sort keys for jumping mover partitions (Band 0/2)."""
    mover_assignments = _cached_mover_assignments(
        edge, mover_partitions, mover_weight_policy
    )
    for jumping_partition in mover_partitions:
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


def _movement_counts_by_role(
    before: Tuple[str, ...],
    after: List[str],
    anchor_taxa: set[str],
    mover_taxa: set[str],
) -> Dict[str, int]:
    """Count taxa whose side-local preprocessing position changed."""
    before_index = {taxon: index for index, taxon in enumerate(before)}
    counts = {"anchor": 0, "mover": 0, "fallback": 0}
    for index, taxon in enumerate(after):
        if before_index.get(taxon) == index:
            continue
        if taxon in mover_taxa:
            counts["mover"] += 1
        elif taxon in anchor_taxa:
            counts["anchor"] += 1
        else:
            counts["fallback"] += 1
    return counts


def _cross_pair_movement_counts_by_role(
    source_after: List[str],
    destination_after: List[str],
    anchor_taxa: set[str],
    mover_taxa: set[str],
) -> Dict[str, int]:
    """Count taxa whose final source/destination positions still differ."""
    destination_index = {taxon: index for index, taxon in enumerate(destination_after)}
    counts = {"anchor": 0, "mover": 0, "fallback": 0}
    for index, taxon in enumerate(source_after):
        if destination_index.get(taxon) == index:
            continue
        if taxon in mover_taxa:
            counts["mover"] += 1
        elif taxon in anchor_taxa:
            counts["anchor"] += 1
        else:
            counts["fallback"] += 1
    return counts


def _handle_circular_rotation(
    edge: Partition,
    sorted_src_taxa: List[str],
    sorted_dest_taxa: List[str],
    mover_partitions: List[Partition],
    src_taxon_sort_key: Dict[str, Tuple[int, int, int]],
    dst_taxon_sort_key: Dict[str, Tuple[int, int, int]],
    circular_boundary_policy: str,
) -> Tuple[List[str], List[str], int]:
    """Apply circular rotation to the final permutations if needed."""
    edge_key = tuple(edge.indices)
    if circular_boundary_policy == "largest_mover_at_zero":
        src_cut_candidate = _boundary_largest_mover_at_zero(
            sorted_src_taxa, mover_partitions
        )
        dst_cut_candidate = _boundary_largest_mover_at_zero(
            sorted_dest_taxa, mover_partitions
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
) -> None:
    """
    Derives and applies leaf orderings for all differing edges between two trees.

    1. Calculates solution mappings if not provided.
    2. Applies `blocked_order_and_apply` to each differing edge.
    3. Aligns root-level common blocks only after actual differing-edge work.
    """
    t_total_start = time.perf_counter()
    if mappings_t1 is None or mappings_t2 is None:
        t_mappings_start = time.perf_counter()
        mappings_t1, mappings_t2 = _get_solution_mappings(
            t1, t2, precomputed_solution=precomputed_solution
        )
        logger.info(
            "[PhaseTimer] anchor_derive_solution_mappings edges=%d %.3fs",
            len(mappings_t1),
            time.perf_counter() - t_mappings_start,
        )

    if not jt_logger.disabled:
        jt_logger.info("Source maps + derived jumping taxa per edge:")

    # Sort pivot edges topologically (subsets before supersets) for correct processing order
    t_sort_start = time.perf_counter()
    ordered_edges = topological_sort_edges(list(mappings_t1.keys()), t1)
    logger.info(
        "[PhaseTimer] anchor_derive_topological_sort edges=%d %.3fs",
        len(ordered_edges),
        time.perf_counter() - t_sort_start,
    )

    # Apply ordering for differing edges in topological order
    t_edges_start = time.perf_counter()
    root_alignment_elapsed = 0.0
    root_partition: Optional[Partition] = None
    root_anchor_nodes: Optional[List[Node]] = None
    root_mover_partitions: Optional[List[Partition]] = None
    if ordered_edges:
        all_taxa_indices = tuple(sorted(t1.taxa_encoding.values()))
        root_partition = Partition(all_taxa_indices, t1.taxa_encoding)
        root_anchor_nodes, root_mover_partitions = _get_stable_anchor_nodes_and_movers(
            root_partition,
            t1,
            t2,
            {},
            {},
            t1,
            common_splits=common_splits,
        )

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

        # Align root-level common blocks after each differing edge. If there are
        # no differing edges, preserve each tree's existing visual order.
        t_root_align_start = time.perf_counter()
        assert root_partition is not None
        blocked_order_and_apply(
            root_partition,
            {},  # No solution-to-source mappings
            {},  # No solution-to-destination mappings
            t1,
            t2,
            mover_weight_policy=mover_weight_policy,
            anchor_weight_policy=anchor_weight_policy,
            circular=circular,
            circular_boundary_policy=circular_boundary_policy,
            common_splits=common_splits,
            precomputed_anchor_nodes=root_anchor_nodes,
            precomputed_mover_partitions=root_mover_partitions,
        )
        root_alignment_elapsed += time.perf_counter() - t_root_align_start
    logger.info(
        "[PhaseTimer] anchor_derive_blocked_edges edges=%d %.3fs",
        len(ordered_edges),
        time.perf_counter() - t_edges_start,
    )
    logger.info(
        "[PhaseTimer] anchor_derive_root_alignment calls=%d %.3fs",
        len(ordered_edges),
        root_alignment_elapsed,
    )
    logger.info(
        "[PhaseTimer] anchor_derive_total edges=%d %.3fs",
        len(ordered_edges),
        time.perf_counter() - t_total_start,
    )


def blocked_order_and_apply(
    edge: Partition,
    solution_to_source: Dict[Partition, Partition],
    solution_to_destination: Dict[Partition, Partition],
    t1: Node,
    t2: Node,
    mover_weight_policy: str = "decreasing",
    anchor_weight_policy: str = "preserve_source",
    circular: bool = False,
    circular_boundary_policy: str = "largest_mover_at_zero",
    common_splits: Optional[PartitionSet[Partition]] = None,
    precomputed_anchor_nodes: Optional[List[Node]] = None,
    precomputed_mover_partitions: Optional[List[Partition]] = None,
) -> None:
    """
    Derive and apply a 3-band leaf order to the subtrees defined by an edge.

    1. Identifies stable common subtrees ("Anchors").
    2. Identifies jumping taxa ("Movers").
    3. Assigns Anchors to the center (Band 1).
    4. Assigns Movers to the extremes (Band 0 or 2).
    5. Reorders `t1` and `t2` in-place to match this visual structure.
    """

    t_total_start = time.perf_counter()
    edge_size = len(edge.indices)
    t_lookup_start = time.perf_counter()
    src_node = t1.find_node_by_split(edge)
    dst_node = t2.find_node_by_split(edge)
    lookup_elapsed = time.perf_counter() - t_lookup_start

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
    t_blocks_start = time.perf_counter()
    stable_anchor_blocks, mover_partitions = _get_stable_anchor_blocks_and_movers(
        edge,
        src_node,
        dst_node,
        solution_to_source,
        solution_to_destination,
        t1,
        common_splits=common_splits,
        precomputed_anchor_nodes=precomputed_anchor_nodes,
        precomputed_mover_partitions=precomputed_mover_partitions,
    )
    anchor_taxa = {taxon for block in stable_anchor_blocks for taxon in block}
    mover_taxa = {taxon for partition in mover_partitions for taxon in partition.taxa}
    blocks_elapsed = time.perf_counter() - t_blocks_start

    # Tuple-based sort keys per taxon.
    # Key = (band, anchor_pos_or_rank, within_block_pos)
    # Bands: 0 = Left Mover, 1 = Anchor, 2 = Right Mover
    src_taxon_sort_key: Dict[str, Tuple[int, int, int]] = {}
    dst_taxon_sort_key: Dict[str, Tuple[int, int, int]] = {}

    t_keys_start = time.perf_counter()
    _assign_anchor_keys(
        stable_anchor_blocks,
        source_index,
        destination_index,
        anchor_weight_policy,
        src_taxon_sort_key,
        dst_taxon_sort_key,
    )

    # Assign banded tuple keys to jumping partitions using deterministic alternation
    _assign_mover_keys(
        edge,
        mover_partitions,
        mover_weight_policy,
        source_index,
        destination_index,
        src_taxon_sort_key,
        dst_taxon_sort_key,
    )
    keys_elapsed = time.perf_counter() - t_keys_start

    # Build final taxa lists and sort using tuple keys. Unhandled taxa are not
    # movers, so treat them as singleton anchors under the same anchor policy
    # instead of letting them drift side-locally or clump into band position 0.
    t_sort_start = time.perf_counter()
    src_taxa_in_edge = list(src_current_order)
    dst_taxa_in_edge = list(dst_current_order)

    def fallback_anchor_key(taxon: str) -> Tuple[int, int, int]:
        anchor_pos = (
            source_index[taxon]
            if anchor_weight_policy == "preserve_source"
            else destination_index[taxon]
        )
        return (1, anchor_pos, 0)

    sorted_src_taxa = sorted(
        src_taxa_in_edge,
        key=lambda t: src_taxon_sort_key.get(t, fallback_anchor_key(t)),
    )
    sorted_dest_taxa = sorted(
        dst_taxa_in_edge,
        key=lambda t: dst_taxon_sort_key.get(t, fallback_anchor_key(t)),
    )
    sort_elapsed = time.perf_counter() - t_sort_start

    # Optional circular rotation of the final permutations for circular rendering
    src_cut = 0
    rotate_elapsed = 0.0
    if circular:
        t_rotate_start = time.perf_counter()
        sorted_src_taxa, sorted_dest_taxa, src_cut = _handle_circular_rotation(
            edge,
            sorted_src_taxa,
            sorted_dest_taxa,
            mover_partitions,
            src_taxon_sort_key,
            dst_taxon_sort_key,
            circular_boundary_policy,
        )
        rotate_elapsed = time.perf_counter() - t_rotate_start

    if not jt_logger.disabled:
        jt_logger.info(
            f"DEBUG: sorted_src_taxa sample: {sorted_src_taxa[:5]} ... {sorted_src_taxa[-5:]}"
        )
        jt_logger.info(f"DEBUG: src_cut={src_cut if circular else 'N/A'}")

    t_reorder_start = time.perf_counter()
    src_reordered = tuple(sorted_src_taxa) != src_current_order
    dst_reordered = tuple(sorted_dest_taxa) != dst_current_order
    if src_reordered:
        src_node.reorder_taxa(sorted_src_taxa, ReorderStrategy.MINIMUM)
    if dst_reordered:
        dst_node.reorder_taxa(sorted_dest_taxa, ReorderStrategy.MINIMUM)
    reorder_elapsed = time.perf_counter() - t_reorder_start
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "[AnchorMovement] edge_size=%d anchors=%d movers=%d fallback=%d "
            "src_moved=%s dst_moved=%s cross_pair_moved=%s",
            edge_size,
            len(anchor_taxa),
            len(mover_taxa),
            len(set(src_current_order) - anchor_taxa - mover_taxa),
            _movement_counts_by_role(
                src_current_order,
                sorted_src_taxa,
                anchor_taxa,
                mover_taxa,
            ),
            _movement_counts_by_role(
                dst_current_order,
                sorted_dest_taxa,
                anchor_taxa,
                mover_taxa,
            ),
            _cross_pair_movement_counts_by_role(
                sorted_src_taxa,
                sorted_dest_taxa,
                anchor_taxa,
                mover_taxa,
            ),
        )
    logger.info(
        "[PhaseTimer] anchor_blocked_order edge_size=%d anchors=%d movers=%d taxa=%d "
        "lookup=%.3fs blocks=%.3fs keys=%.3fs sort=%.3fs rotate=%.3fs reorder=%.3fs "
        "src_changed=%s dst_changed=%s total=%.3fs",
        edge_size,
        len(stable_anchor_blocks),
        len(mover_partitions),
        len(src_taxa_in_edge),
        lookup_elapsed,
        blocks_elapsed,
        keys_elapsed,
        sort_elapsed,
        rotate_elapsed,
        reorder_elapsed,
        src_reordered,
        dst_reordered,
        time.perf_counter() - t_total_start,
    )
