#!/usr/bin/env python
"""Analyze what the correct leaf order should be after Ostrich moves."""
import sys
import os
sys.path.insert(0, os.getcwd())

from brancharchitect.parser.newick_parser import parse_newick
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
    dest_order = list(dest.get_current_order())

    print("=== Current Orders ===")
    print(f"Source: {source_order}")
    print(f"Dest:   {dest_order}")

    # The key insight: if we want ONLY Ostrich to move visually,
    # then the destination tree should be reordered to match source order
    # EXCEPT for Ostrich's new position

    print("\n=== Analysis ===")
    print("Source order (without Ostrich):")
    source_no_ostrich = [t for t in source_order if t != 'Ostrich']
    print(f"  {source_no_ostrich}")

    print("\nDest order (without Ostrich):")
    dest_no_ostrich = [t for t in dest_order if t != 'Ostrich']
    print(f"  {dest_no_ostrich}")

    print("\n=== Key Question ===")
    print("Are the non-Ostrich taxa in the same relative order?")
    print(f"Same order: {source_no_ostrich == dest_no_ostrich}")

    if source_no_ostrich != dest_no_ostrich:
        print("\nDifferences:")
        for i, (s, d) in enumerate(zip(source_no_ostrich, dest_no_ostrich)):
            if s != d:
                print(f"  Position {i}: source='{s}' vs dest='{d}'")

    print("\n=== What Should Happen ===")
    print("If we want ONLY Ostrich to move visually:")
    print("1. The destination tree should be reordered to match source's non-Ostrich order")
    print("2. Then Ostrich moves from position 11 to position 1")
    print("3. The final order should be:")

    # Build the expected order: Ostrich at position 1, rest in source order
    expected = ['Emu', 'Ostrich'] + source_no_ostrich[1:]  # Skip Emu which is already at 0
    print(f"   {expected}")

    print("\n=== Current Problem ===")
    print("The destination tree has a DIFFERENT internal order for non-Ostrich taxa.")
    print("This is because the Newick string encodes a specific traversal order.")
    print("The reordering step should align the destination to source order BEFORE moving Ostrich.")


if __name__ == "__main__":
    main()
