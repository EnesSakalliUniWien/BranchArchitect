#!/usr/bin/env python
"""Test what happens if we align destination to source order before interpolation."""
import sys
import os
sys.path.insert(0, os.getcwd())

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.tree_interpolation.sequential_interpolation import SequentialInterpolationBuilder
from brancharchitect.logger import jt_logger

def main():
    jt_logger.disabled = True

    # Bird trees
    source_newick = "(Emu,(((((((BrushTurkey,Chicken),(magpiegoose,duck)),((LBPenguin,GaviaStellata),(oystercatcher,turnstone))),(GreatRhea,LesserRhea)),Ostrich),(((lbmoa,EasternMoa),Dinornis),((Alligator,Caiman),(ECtinamou,(Gtinamou,Crypturellus))))),(BrownKiwi,(LSKiwi,gskiwi))),Cassowary);"
    dest_newick = "(Emu,((Ostrich,((((lbmoa,EasternMoa),Dinornis),((Alligator,Caiman),(ECtinamou,(Gtinamou,Crypturellus)))),((((BrushTurkey,Chicken),(magpiegoose,duck)),((LBPenguin,GaviaStellata),(oystercatcher,turnstone))),(GreatRhea,LesserRhea)))),(BrownKiwi,(LSKiwi,gskiwi))),Cassowary);"

    source = parse_newick(source_newick)
    dest = parse_newick(dest_newick)

    # Ensure same encoding
    dest.initialize_split_indices(source.taxa_encoding)

    source_order = list(source.get_current_order())
    dest_order_original = list(dest.get_current_order())

    print("=== Original Orders ===")
    print(f"Source: {source_order}")
    print(f"Dest (original): {dest_order_original}")

    # Now align destination to source order
    print("\n=== Aligning Destination to Source Order ===")
    dest_aligned = dest.deep_copy()
    dest_aligned.reorder_taxa(source_order)
    dest_order_aligned = list(dest_aligned.get_current_order())
    print(f"Dest (aligned): {dest_order_aligned}")
    print(f"Aligned matches source: {dest_order_aligned == source_order}")

    # Verify topology is preserved
    dest_splits = set(dest.to_weighted_splits().keys())
    dest_aligned_splits = set(dest_aligned.to_weighted_splits().keys())
    print(f"Topology preserved: {dest_splits == dest_aligned_splits}")

    # Run interpolation with aligned destination
    print("\n=== Running Interpolation with ALIGNED Destination ===")
    builder = SequentialInterpolationBuilder()
    result = builder.build([source.deep_copy(), dest_aligned.deep_copy()])

    print(f"Total trees: {len(result.interpolated_trees)}")
    print(f"Pair counts: {result.pair_interpolated_tree_counts}")

    final_order = list(result.interpolated_trees[-1].get_current_order())
    print(f"\nFinal order: {final_order}")
    print(f"Matches aligned dest: {final_order == dest_order_aligned}")

    # Check what moved
    print("\n=== Movement Analysis ===")
    for i, tree in enumerate(result.interpolated_trees):
        subtree = result.current_subtree_tracking[i]
        if subtree:
            print(f"Tree {i}: Moving {sorted(list(subtree.taxa))}")

    # Compare with original (unaligned) interpolation
    print("\n=== Running Interpolation with ORIGINAL Destination ===")
    result_orig = builder.build([source.deep_copy(), dest.deep_copy()])

    final_order_orig = list(result_orig.interpolated_trees[-1].get_current_order())
    print(f"Final order (original): {final_order_orig}")
    print(f"Matches original dest: {final_order_orig == dest_order_original}")

    print("\n=== Comparison ===")
    print(f"Aligned final == Original final: {final_order == final_order_orig}")


if __name__ == "__main__":
    main()
