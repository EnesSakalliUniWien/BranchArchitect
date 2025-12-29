#!/usr/bin/env python
"""
Debug script to analyze tree reordering between trees 3-4 (0-indexed).
Focuses on understanding why Ostrich, LesserRhea, and GreatRhea are moving unexpectedly.
"""

from brancharchitect.io import read_newick
from brancharchitect.movie_pipeline.types import PipelineConfig
from brancharchitect.movie_pipeline.tree_interpolation_pipeline import (
    TreeInterpolationPipeline,
)
from brancharchitect.jumping_taxa.lattice.solvers.lattice_solver import LatticeSolver
from brancharchitect.tree_interpolation.subtree_paths.execution.reordering import (
    reorder_tree_toward_destination,
)
from brancharchitect.tree_interpolation.subtree_paths.planning.builder import (
    build_edge_plan,
)
from brancharchitect.tree_interpolation.subtree_paths.analysis.split_analysis import (
    get_unique_splits_for_current_pivot_edge_subtree,
)


def print_tree_order(tree, label):
    """Print the leaf order of a tree."""
    order = tree.get_current_order()
    print(f"\n{label} leaf order ({len(order)} taxa):")
    for i, taxon in enumerate(order):
        print(f"  {i:2d}: {taxon}")
    return order


def analyze_jumping_taxa(source, dest, pair_idx):
    """Analyze jumping taxa between two trees."""
    print(f"\n{'=' * 80}")
    print(f"ANALYZING TREE PAIR {pair_idx} -> {pair_idx + 1}")
    print(f"{'=' * 80}")

    # Get leaf orders
    source_order = print_tree_order(source, f"Tree {pair_idx} (Source)")
    dest_order = print_tree_order(dest, f"Tree {pair_idx + 1} (Destination)")

    # Find taxa that changed position
    print(f"\n--- Position Changes ---")
    source_positions = {taxon: i for i, taxon in enumerate(source_order)}
    dest_positions = {taxon: i for i, taxon in enumerate(dest_order)}

    changes = []
    for taxon in source_order:
        src_pos = source_positions[taxon]
        dst_pos = dest_positions[taxon]
        if src_pos != dst_pos:
            changes.append((taxon, src_pos, dst_pos, dst_pos - src_pos))

    if changes:
        print(f"Taxa that changed position:")
        for taxon, src_pos, dst_pos, delta in sorted(
            changes, key=lambda x: abs(x[3]), reverse=True
        ):
            print(f"  {taxon}: {src_pos} -> {dst_pos} (delta: {delta:+d})")
    else:
        print("No position changes detected")

    # Run lattice solver to find jumping taxa
    print(f"\n--- Lattice Solver Results ---")
    try:
        solver = LatticeSolver(source, dest)
        solution_dict, _ = solver.solve_iteratively()

        if solution_dict:
            print(f"Found {len(solution_dict)} pivot edges with jumping taxa:")
            for pivot_edge, jumping_taxa_list in solution_dict.items():
                pivot_taxa = (
                    pivot_edge.taxa if hasattr(pivot_edge, "taxa") else str(pivot_edge)
                )
                print(f"\n  Pivot edge: {pivot_taxa}")
                for jt in jumping_taxa_list:
                    jt_taxa = jt.taxa if hasattr(jt, "taxa") else str(jt)
                    print(f"    Jumping taxa: {jt_taxa}")
        else:
            print("No jumping taxa found (trees may be identical)")

        return solution_dict
    except Exception as e:
        print(f"Error running lattice solver: {e}")
        import traceback

        traceback.print_exc()
        return None


def debug_reordering_step(source, dest, pivot_edge, moving_subtree, step_name):
    """Debug a single reordering step."""
    print(f"\n--- Reordering Debug: {step_name} ---")
    print(
        f"Pivot edge: {pivot_edge.taxa if hasattr(pivot_edge, 'taxa') else pivot_edge}"
    )
    print(
        f"Moving subtree: {moving_subtree.taxa if hasattr(moving_subtree, 'taxa') else moving_subtree}"
    )

    # Find the subtrees
    source_subtree = source.find_node_by_split(pivot_edge)
    dest_subtree = dest.find_node_by_split(pivot_edge)

    if source_subtree is None:
        print(f"  ERROR: Pivot edge not found in source tree!")
        return None
    if dest_subtree is None:
        print(f"  ERROR: Pivot edge not found in destination tree!")
        return None

    source_order = list(source_subtree.get_current_order())
    dest_order = list(dest_subtree.get_current_order())
    mover_leaves = (
        set(moving_subtree.taxa) if hasattr(moving_subtree, "taxa") else set()
    )

    print(f"\n  Source subtree order: {source_order}")
    print(f"  Dest subtree order:   {dest_order}")
    print(f"  Mover leaves:         {mover_leaves}")

    # Compute anchors
    source_anchors = [t for t in source_order if t not in mover_leaves]
    print(f"  Source anchors:       {source_anchors}")

    # Find anchors before movers in destination
    anchors_before_dest = []
    for taxon in dest_order:
        if taxon in mover_leaves:
            break
        anchors_before_dest.append(taxon)
    print(f"  Anchors before movers in dest: {anchors_before_dest}")

    # Compute insertion index
    anchor_rank_src = {a: i for i, a in enumerate(source_anchors)}
    anchors_before_in_src_order = sorted(
        (a for a in anchors_before_dest if a in anchor_rank_src),
        key=lambda a: anchor_rank_src[a],
    )
    insertion_index = len(anchors_before_in_src_order)
    print(f"  Anchors before (in src order): {anchors_before_in_src_order}")
    print(f"  Insertion index: {insertion_index}")

    # Build new order
    new_order = list(source_anchors)
    mover_block_src = [t for t in source_order if t in mover_leaves]
    print(f"  Mover block (src order): {mover_block_src}")

    new_order[insertion_index:insertion_index] = mover_block_src
    print(f"  New order after insertion: {new_order}")

    # Apply reordering
    try:
        result = reorder_tree_toward_destination(
            source_tree=source,
            destination_tree=dest,
            current_pivot_edge=pivot_edge,
            moving_subtree_partition=moving_subtree,
        )
        result_order = list(result.get_current_order())
        print(f"  Result tree order: {result_order}")
        return result
    except Exception as e:
        print(f"  ERROR during reordering: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    # Load trees from the file
    tree_file = "small_example copy 3.tree"
    print(f"Loading trees from: {tree_file}")

    trees = read_newick(tree_file, treat_zero_as_epsilon=True)
    print(f"Loaded {len(trees)} trees")

    # Focus on trees 3-4 (0-indexed)
    pair_idx = 3
    if pair_idx + 1 >= len(trees):
        print(f"Error: Not enough trees. Need at least {pair_idx + 2} trees.")
        return

    source = trees[pair_idx]
    dest = trees[pair_idx + 1]

    # Analyze the pair
    solution_dict = analyze_jumping_taxa(source, dest, pair_idx)

    if solution_dict:
        print(f"\n{'=' * 80}")
        print("DETAILED REORDERING ANALYSIS")
        print(f"{'=' * 80}")

        for pivot_edge, jumping_taxa_list in solution_dict.items():
            for i, moving_subtree in enumerate(jumping_taxa_list):
                debug_reordering_step(
                    source,
                    dest,
                    pivot_edge,
                    moving_subtree,
                    f"Pivot {pivot_edge.taxa}, JT {i + 1}",
                )

        # Debug the builder selection order
        print(f"\n{'=' * 80}")
        print("BUILDER SELECTION ORDER ANALYSIS")
        print(f"{'=' * 80}")

        # Get the pivot edge (there's only one)
        pivot_edge = list(solution_dict.keys())[0]
        jumping_taxa_list = solution_dict[pivot_edge]

        # Build collapse/expand splits by subtree (mimicking what the pipeline does)
        from brancharchitect.elements.partition_set import PartitionSet

        collapse_splits_by_subtree = {}
        expand_splits_by_subtree = {}

        # Get all splits
        all_collapse, all_expand = get_unique_splits_for_current_pivot_edge_subtree(
            source, dest, pivot_edge
        )

        print(f"\nAll collapse splits: {len(all_collapse)}")
        print(f"All expand splits: {len(all_expand)}")

        # Show jumping taxa and their sizes
        print(f"\nJumping taxa groups:")
        for i, jt in enumerate(jumping_taxa_list):
            print(f"  {i + 1}. {jt.taxa} ({len(jt.taxa)} taxa)")

        # Build the edge plan to see selection order
        print(f"\nBuilding edge plan...")

        # We need to set up the splits_by_subtree dicts
        # For now, let's just assign each jumping taxa to its own expand splits
        for jt in jumping_taxa_list:
            # Find expand splits that contain this jumping taxa
            jt_indices = set(jt.indices)
            matching_expands = PartitionSet(encoding=pivot_edge.encoding)
            for split in all_expand:
                if jt_indices.issubset(set(split.indices)):
                    matching_expands.add(split)
            expand_splits_by_subtree[jt] = matching_expands
            collapse_splits_by_subtree[jt] = PartitionSet(encoding=pivot_edge.encoding)
            print(f"  JT {jt.taxa}: {len(matching_expands)} expand splits")

        # Now build the plan
        plans = build_edge_plan(
            expand_splits_by_subtree,
            collapse_splits_by_subtree,
            source,
            dest,
            pivot_edge,
        )

        print(f"\nExecution order from builder:")
        for i, (subtree, plan) in enumerate(plans.items()):
            print(f"  {i + 1}. {subtree.taxa}")
            print(f"      Collapse: {len(plan['collapse']['path_segment'])} splits")
            print(f"      Expand: {len(plan['expand']['path_segment'])} splits")

    # Print newick strings for source and destination
    print(f"\n{'=' * 80}")
    print("NEWICK STRINGS")
    print(f"{'=' * 80}")

    print(f"\nSource Tree {pair_idx} (newick):")
    print(source.to_newick(lengths=False))

    print(f"\nDestination Tree {pair_idx + 1} (newick):")
    print(dest.to_newick(lengths=False))

    # Also run the full pipeline to see the interpolated result
    print(f"\n{'=' * 80}")
    print("FULL PIPELINE EXECUTION (trees 3-4 only)")
    print(f"{'=' * 80}")

    config = PipelineConfig(
        enable_rooting=False,
        bidirectional_optimization=False,
        use_anchor_ordering=True,
        anchor_weight_policy="destination",
        circular=False,
        enable_debug_visualization=True,
    )

    # Process just the pair
    pair_trees = [source.deep_copy(), dest.deep_copy()]
    pipeline = TreeInterpolationPipeline(config=config)

    try:
        result = pipeline.process_trees(trees=pair_trees)

        print(f"\nInterpolated {len(result['interpolated_trees'])} trees")

        # Show the order of each interpolated tree
        for i, tree in enumerate(result["interpolated_trees"]):
            order = tree.get_current_order()
            print(f"\nInterpolated tree {i}:")
            # Just show key taxa positions
            key_taxa = ["Ostrich", "GreatRhea", "LesserRhea"]
            for taxon in key_taxa:
                if taxon in order:
                    pos = order.index(taxon)
                    print(f"  {taxon}: position {pos}")

            # Print newick string for this interpolated tree
            print(f"  Newick: {tree.to_newick(lengths=False)}")

    except Exception as e:
        print(f"Pipeline error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
