"""Utilities for building and solving per-edge ordering CSPs.

This module contains the reusable pieces that were previously embedded in
``test_all_edges_block_orders.py``.  Functions here can be imported by scripts,
CLI tools, or notebooks in order to analyse edge-specific orderings without
having to duplicate the CSP setup code.
"""

from __future__ import annotations
from typing import Dict, Set
from brancharchitect.elements.partition import Partition
from brancharchitect.tree import Node
from brancharchitect.jumping_taxa.lattice.mapping import map_solutions_to_atoms
from brancharchitect.jumping_taxa.lattice.iterate_lattice_algorithm import (
    iterate_lattice_algorithm,
)

__all__ = [
    "derive_and_apply_order",
]

# ---------------------------------------------------------------------------
# Edge analysis helpers
# ---------------------------------------------------------------------------


def derive_and_apply_order(
    edge: Partition,
    sources: Dict[Partition, Partition],
    destinations: Dict[Partition, Partition],
    t1: Node,
    t2: Node,
):
    """
    Derive and apply a new leaf order to the subtrees defined by an edge.

    This function calculates a new ordering for the taxa in the subtrees
    (rooted at `src_node` and `dst_node`) corresponding to the given `edge`.
    The ordering is designed to group moving partitions and place them at the
    extremes of the ordering, with stable "anchor" taxa in between.

    The `t1` and `t2` nodes are modified in-place.
    """

    # This variable was unused.
    # source_dest_taxa: Set[str] = set()
    # for partition in list(sources.values()) + list(destinations.values()):
    #     source_dest_taxa.update(partition.taxa)

    src_node = t1.find_node_by_split(edge)
    dst_node = t2.find_node_by_split(edge)

    src_index = {taxon: i for i, taxon in enumerate(dst_node.get_current_order())}
    destination_index = {
        taxon: i for i, taxon in enumerate(dst_node.get_current_order())
    }
    source_blocked = []

    common_splits = src_node.to_splits().intersection(dst_node.to_splits()) - {edge}

    common_splits = common_splits.atom()

    # The logic below assigns large positive or negative weights to taxa
    # to force a specific sorting order.
    # - Moving taxa are pushed to the ends.
    # - Taxa that are part of the source/destination of a move but not the
    #   mover itself are grouped with the movers.

    # Create a unified list of all stable blocks (common splits and free leaves).
    # This is a more elegant way to represent all the non-moving parts.
    taxa_in_common_blocks = {taxon for s in common_splits for taxon in s.taxa}
    free_taxa = set(edge.taxa) - taxa_in_common_blocks

    # Get the leaf order for each common split as a block.
    source_blocked = [
        t1.find_node_by_split(cs).get_current_order() for cs in common_splits
    ]
    # Add each free taxon as its own block (a single-element tuple).
    source_blocked.extend([(taxon,) for taxon in sorted(list(free_taxa))])

    print("Source blocked (common splits + free taxa):")
    print(source_blocked)

    # Note: The 'destination_blocked' list was identical and is redundant.
    # The 'source_blocked' list now contains all stable blocks but is not
    # used later in this function's current logic.
    # REMOVE THIS LINE: destination_index = {}

    src_block_weights = {}
    destination_block_weights = {}

    for i, block in enumerate(source_blocked):
        # `block` is already a tuple, e.g., ('F',).
        # We ensure it's sorted for canonical representation.
        canonical_key = tuple(sorted(block))
        src_block_weights[canonical_key] = destination_index.get(block[0], 0)
        destination_block_weights[canonical_key] = destination_index.get(block[0], 0)

    for i, moving_partition in enumerate(destinations.keys()):
        weight = 100 ** (i + 2)

        # Convert the Partition's frozenset of taxa to a sorted tuple.
        canonical_key = tuple(sorted(moving_partition.taxa))

        src_block_weights[canonical_key] = -weight
        destination_block_weights[canonical_key] = weight

        # When iterating, also use the canonical form for lookups.
        for i, block_key in enumerate(src_block_weights.keys()):
            if set(block_key) & set(destinations[moving_partition].taxa) and set(
                block_key
            ) & set(moving_partition.taxa):
                src_block_weights[block_key] = -weight + 1

        for i, block_key in enumerate(destination_block_weights.keys()):
            if set(block_key) & set(destinations[moving_partition].taxa) and set(
                block_key
            ) & set(moving_partition.taxa):
                destination_block_weights[block_key] = weight - 1

    print(destination_block_weights)
    print(src_block_weights)

    print(f"Source index for edge {edge}: {src_index}")
    print(f"Destination index for edge {edge}: {destination_index}")

    sorted_src_taxa = sorted(src_index.keys(), key=lambda taxon: src_index[taxon])

    sorted_dest_taxa = sorted(
        destination_index.keys(), key=lambda taxon: destination_index[taxon]
    )

    print(f"Sorted Taxa {edge}: {sorted_src_taxa}")
    print(f"Sorted Taxa {edge}: {sorted_dest_taxa}")

    src_node.reorder_taxa(sorted_src_taxa)
    print(f"New order for T1 at edge {edge}: {src_node.to_newick(lengths=False)}")
    dst_node.reorder_taxa(sorted_dest_taxa)
    print(f"New order for T2 at edge {edge}: {dst_node.to_newick(lengths=False)}")


def _get_solution_mappings(t1: Node, t2: Node) -> tuple[dict, dict]:
    """
    Calculates the solution mappings for moving partitions between two trees.
    """
    solutions = iterate_lattice_algorithm(input_tree1=t1, input_tree2=t2)
    unique_splits_t1 = t1.to_splits() - t2.to_splits()
    unique_splits_t2 = t2.to_splits() - t1.to_splits()

    mappings_t1, mappings_t2 = map_solutions_to_atoms(
        solutions,
        unique_splits_t1=unique_splits_t1,
        unique_splits_t2=unique_splits_t2,
    )
    return mappings_t1, mappings_t2


def derive_order_for_pair(
    t1: Node,
    t2: Node,
    mappings_t1: Dict[Partition, Dict[Partition, Partition]] = None,
    mappings_t2: Dict[Partition, Dict[Partition, Partition]] = None,
):
    """
    Derives and applies leaf orderings for all differing edges between two trees.

    If solution mappings are not provided, they are calculated first.
    """
    if mappings_t1 is None or mappings_t2 is None:
        mappings_t1, mappings_t2 = _get_solution_mappings(t1, t2)

    print("Source maps + derived movers per edge:")
    for edge, mapping in mappings_t1.items():
        blocked_order_and_apply(
            edge,
            mapping,
            mappings_t2.get(edge, {}),
            t1,
            t2,
        )


def blocked_order_and_apply(
    edge: Partition,
    sources: Dict[Partition, Partition],
    destinations: Dict[Partition, Partition],
    t1: Node,
    t2: Node,
):
    """
    Derive and apply a new leaf order to the subtrees defined by an edge.

    This function calculates a new ordering for the taxa in the subtrees
    (rooted at `src_node` and `dst_node`) corresponding to the given `edge`.
    The ordering is designed to group moving partitions and place them at the
    extremes of the ordering, with stable "anchor" taxa in between.

    The `t1` and `t2` nodes are modified in-place.
    """

    # This variable was unused.
    # source_dest_taxa: Set[str] = set()
    # for partition in list(sources.values()) + list(destinations.values()):
    #     source_dest_taxa.update(partition.taxa)

    src_node = t1.find_node_by_split(edge)
    dst_node = t2.find_node_by_split(edge)

    # Correctly create indices from both source and destination trees
    src_index = {taxon: i for i, taxon in enumerate(src_node.get_current_order())}
    destination_index = {
        taxon: i for i, taxon in enumerate(dst_node.get_current_order())
    }

    source_blocked = []

    common_splits = src_node.to_splits().intersection(dst_node.to_splits()) - {edge}

    common_splits = common_splits.atom()

    # Create a unified list of all stable blocks (common splits and free leaves).
    # This is a more elegant way to represent all the non-moving parts.
    taxa_in_common_blocks = {taxon for s in common_splits for taxon in s.taxa}
    free_taxa = set(edge.taxa) - taxa_in_common_blocks

    # Get the leaf order for each common split as a block.
    source_blocked = [
        t1.find_node_by_split(cs).get_current_order() for cs in common_splits
    ]
    # Add each free taxon as its own block (a single-element tuple).
    source_blocked.extend([(taxon,) for taxon in sorted(list(free_taxa))])

    print("Source blocked (common splits + free taxa):")
    print(source_blocked)

    # Note: The 'destination_blocked' list was identical and is redundant.
    # The 'source_blocked' list now contains all stable blocks but is not
    # used later in this function's current logic.
    # REMOVE THIS LINE: destination_index = {}

    src_block_weights = {}
    destination_block_weights = {}

    for i, block in enumerate(source_blocked):
        # `block` is already a tuple, e.g., ('F',).
        # We ensure it's sorted for canonical representation.
        canonical_key = tuple(sorted(block))
        # Anchor weights for the source tree's new order are based on the anchor's position in the DESTINATION tree.
        src_block_weights[canonical_key] = destination_index.get(block[0], 0)
        destination_block_weights[canonical_key] = src_index.get(block[0], 0)

    for i, moving_partition in enumerate(destinations.keys()):
        weight = 0
        if i % 2 == 0:
            weight = 100 ** (i + 2)
        else:
            weight = -(100 ** (i + 2))

        mover_key = tuple(sorted(moving_partition.taxa))

        # 1. Assign weights to the mover itself.
        src_block_weights[mover_key] = weight
        destination_block_weights[mover_key] = -weight

    # --- Start: Transform block weights into a final sorted list ---
    # 1. Create new weight dictionaries for individual taxa.
    src_taxon_weights = {}
    dest_taxon_weights = {}

    # 2. Unroll the block weights into taxon weights.
    # For each block, assign its weight to every taxon within it.
    # A small offset is added to preserve the internal order of taxa within a block.
    for block, weight in src_block_weights.items():
        for i, taxon in enumerate(block):
            src_taxon_weights[taxon] = weight + (i * 0.01)

    for block, weight in destination_block_weights.items():
        for i, taxon in enumerate(block):
            dest_taxon_weights[taxon] = weight + (i * 0.01)

    # 3. Sort the taxa based on their final calculated weights.
    sorted_src_taxa = sorted(
        src_taxon_weights.keys(), key=lambda taxon: src_taxon_weights[taxon]
    )
    sorted_dest_taxa = sorted(
        dest_taxon_weights.keys(), key=lambda taxon: dest_taxon_weights[taxon]
    )
    # --- End: Transformation logic ---

    print(f"Sorted Taxa {edge}: {sorted_src_taxa}")
    print(f"Sorted Taxa {edge}: {sorted_dest_taxa}")

    from brancharchitect.tree import ReorderStrategy

    if weight < 0:
        print(f"New order for T1 at edge {edge}: {src_node.to_newick(lengths=False)}")
        print(f"New order for T2 at edge {edge}: {dst_node.to_newick(lengths=False)}")
        src_node.reorder_taxa(sorted_src_taxa, ReorderStrategy.MINIMUM)
        dst_node.reorder_taxa(sorted_dest_taxa, ReorderStrategy.MAXIMUM)
    else:
        print(f"New order for T1 at edge {edge}: {src_node.to_newick(lengths=False)}")
        print(f"New order for T2 at edge {edge}: {dst_node.to_newick(lengths=False)}")
        src_node.reorder_taxa(sorted_src_taxa, ReorderStrategy.MAXIMUM)
        dst_node.reorder_taxa(sorted_dest_taxa, ReorderStrategy.MINIMUM)
