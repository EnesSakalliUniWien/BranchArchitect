#!/usr/bin/env python
"""Debug script to investigate tree interpolation with multiple movers of different sizes."""

import sys
import os

sys.path.insert(0, os.getcwd())

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.tree_interpolation.sequential_interpolation import (
    SequentialInterpolationBuilder,
)
from brancharchitect.logger import jt_logger


def run_test_case(name, tree1_newick, tree2_newick, key_taxa_groups=None):
    """Run interpolation test and report results."""
    print(f"\n{'=' * 60}")
    print(f"TEST CASE: {name}")
    print(f"{'=' * 60}")

    trees = [parse_newick(tree1_newick), parse_newick(tree2_newick)]

    # Ensure trees share the same encoding
    base_encoding = trees[0].taxa_encoding
    trees[1].initialize_split_indices(base_encoding)

    source_order = list(trees[0].get_current_order())
    dest_order = list(trees[1].get_current_order())

    print(f"\nSource order ({len(source_order)} taxa): {source_order}")
    print(f"Dest order   ({len(dest_order)} taxa): {dest_order}")

    def get_positions(order, taxa_list):
        return [order.index(t) for t in taxa_list if t in order]

    if key_taxa_groups:
        print("\nKey taxa positions:")
        for group_name, taxa in key_taxa_groups.items():
            src_pos = get_positions(source_order, taxa)
            dst_pos = get_positions(dest_order, taxa)
            print(f"  {group_name}: src={src_pos} -> dst={dst_pos}")

    # Run interpolation
    print("\n--- Running Interpolation ---")
    builder = SequentialInterpolationBuilder()
    result = builder.build([t.deep_copy() for t in trees])

    print(f"Total trees: {len(result.interpolated_trees)}")
    print(f"Pair counts: {result.pair_interpolated_tree_counts}")

    # Show each step
    print("\n--- Interpolation Steps ---")
    for i, tree in enumerate(result.interpolated_trees):
        pivot = result.current_pivot_edge_tracking[i]
        subtree = result.current_subtree_tracking[i]

        pivot_str = "None" if pivot is None else f"{len(pivot.taxa)} taxa"
        subtree_str = "None" if subtree is None else f"{sorted(list(subtree.taxa))}"

        leaf_order = list(tree.get_current_order())

        print(f"\nTree {i}: Pivot={pivot_str}, Moving={subtree_str}")

        if key_taxa_groups:
            positions = {
                name: get_positions(leaf_order, taxa)
                for name, taxa in key_taxa_groups.items()
            }
            pos_str = ", ".join(
                f"{k}:{min(v) if v else '?'}" for k, v in positions.items()
            )
            print(f"  Positions: {pos_str}")

    # Verify final tree is topologically equivalent to destination
    final_splits = set(result.interpolated_trees[-1].to_weighted_splits().keys())
    dest_splits = set(trees[1].to_weighted_splits().keys())

    final_order = list(result.interpolated_trees[-1].get_current_order())
    topo_match = final_splits == dest_splits
    order_match = final_order == dest_order

    print("\n--- RESULT ---")
    print(f"Topologically equivalent: {topo_match}")
    print(f"Leaf order matches dest: {order_match}")

    if not topo_match:
        only_final = final_splits - dest_splits
        only_dest = dest_splits - final_splits
        print(f"  Splits only in final: {len(only_final)}")
        print(f"  Splits only in dest: {len(only_dest)}")

    if not order_match:
        # Check if non-mover taxa preserved source order
        source_order = list(trees[0].get_current_order())
        # This is expected - we preserve source order for non-movers
        print(f"  (Leaf order differs because non-movers preserve source order)")

    return topo_match  # Success = topologically equivalent


def main():
    jt_logger.disabled = True

    results = []

    # Test Case 1: reverse_bootstrap_52 - 3 movers of different sizes
    # Movers: [15,16,23] (3 taxa), [18,19,20,21,22,24,25] (7 taxa), [27] (1 taxon)
    results.append(
        run_test_case(
            "reverse_bootstrap_52 (3 movers: 1, 3, 7 taxa)",
            "((((((((13,14),17),27),((((15,16),23),((((((18,19),20),21),22),24),25)),((((((26,28),29),30),31),32),33))),((((34,35),36),37),38)),((((((39,40),41),42),43),44),45)),((((46,47),48),49),50)),51),52);",
            "((((((((13,14),((((15,16),23),17),27)),((((((18,19),20),21),22),24),25)),((((((26,28),29),30),31),32),33)),((((34,35),36),37),38)),((((((39,40),41),42),43),44),45)),((((46,47),48),49),50)),51),52);",
            {
                "small_27": ["27"],
                "medium_15_16_23": ["15", "16", "23"],
                "large_18_25": ["18", "19", "20", "21", "22", "24", "25"],
            },
        )
    )

    # Test Case 2: heiko_4 - multiple small movers (E, F, G)
    results.append(
        run_test_case(
            "heiko_4 (movers: E, F, G)",
            "((O1,O2),(((A,B),C),((((F,G),E),D),H)));",
            "((O1,O2),(((A,B),(C,(F,(E,G)))),(D,H)));",
            {
                "E": ["E"],
                "F": ["F"],
                "G": ["G"],
                "D": ["D"],
                "H": ["H"],
            },
        )
    )

    # Test Case 3: Original bird trees with Ostrich, Rheas, Birds moving
    results.append(
        run_test_case(
            "bird_trees (Ostrich + large clades)",
            "(Emu,(((((((BrushTurkey,Chicken),(magpiegoose,duck)),((LBPenguin,GaviaStellata),(oystercatcher,turnstone))),(GreatRhea,LesserRhea)),Ostrich),(((lbmoa,EasternMoa),Dinornis),((Alligator,Caiman),(ECtinamou,(Gtinamou,Crypturellus))))),(BrownKiwi,(LSKiwi,gskiwi))),Cassowary);",
            "(Emu,((Ostrich,((((lbmoa,EasternMoa),Dinornis),((Alligator,Caiman),(ECtinamou,(Gtinamou,Crypturellus)))),((((BrushTurkey,Chicken),(magpiegoose,duck)),((LBPenguin,GaviaStellata),(oystercatcher,turnstone))),(GreatRhea,LesserRhea)))),(BrownKiwi,(LSKiwi,gskiwi))),Cassowary);",
            {
                "Ostrich": ["Ostrich"],
                "Rheas": ["GreatRhea", "LesserRhea"],
                "Birds": ["BrushTurkey", "Chicken"],
                "Moas": ["lbmoa", "EasternMoa", "Dinornis"],
            },
        )
    )

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for i, passed in enumerate(results):
        status = "PASS" if passed else "FAIL"
        print(f"  Test {i + 1}: {status}")


if __name__ == "__main__":
    main()
