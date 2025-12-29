#!/usr/bin/env python
"""Test alignment with heavier interpolation cases (multiple movers)."""
import sys
import os
sys.path.insert(0, os.getcwd())

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.tree_interpolation.sequential_interpolation import SequentialInterpolationBuilder
from brancharchitect.leaforder.pairwise_alignment import final_pairwise_alignment_pass
from brancharchitect.logger import jt_logger

def test_case(name, tree1_newick, tree2_newick, key_taxa_groups=None):
    """Test interpolation with and without pre-alignment."""
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"{'='*70}")

    source = parse_newick(tree1_newick)
    dest = parse_newick(tree2_newick)
    dest.initialize_split_indices(source.taxa_encoding)

    source_order = list(source.get_current_order())
    dest_order = list(dest.get_current_order())

    print(f"\nSource order: {source_order[:10]}..." if len(source_order) > 10 else f"\nSource order: {source_order}")
    print(f"Dest order:   {dest_order[:10]}..." if len(dest_order) > 10 else f"Dest order:   {dest_order}")

    # Topology analysis
    source_splits = set(source.to_weighted_splits().keys())
    dest_splits = set(dest.to_weighted_splits().keys())
    common = source_splits & dest_splits
    only_source = source_splits - dest_splits
    only_dest = dest_splits - source_splits

    print(f"\nTopology: {len(source_splits)} source splits, {len(dest_splits)} dest splits")
    print(f"  Common: {len(common)}, Only source: {len(only_source)}, Only dest: {len(only_dest)}")

    # Test 1: Original (no pre-alignment)
    print("\n--- Test 1: WITHOUT pre-alignment ---")
    builder = SequentialInterpolationBuilder()
    result1 = builder.build([source.deep_copy(), dest.deep_copy()])

    final1 = list(result1.interpolated_trees[-1].get_current_order())
    final1_splits = set(result1.interpolated_trees[-1].to_weighted_splits().keys())
    topo_match1 = final1_splits == dest_splits
    order_match1 = final1 == dest_order

    movers1 = set()
    for subtree in result1.current_subtree_tracking:
        if subtree:
            movers1.update(subtree.taxa)

    print(f"  Movers detected: {sorted(movers1)}")
    print(f"  Total trees: {len(result1.interpolated_trees)}")
    print(f"  Topology match: {topo_match1}")
    print(f"  Order match: {order_match1}")

    # Test 2: With pre-alignment of destination to source
    print("\n--- Test 2: WITH pre-alignment (dest → source order) ---")
    dest_aligned = dest.deep_copy()
    dest_aligned.reorder_taxa(source_order)
    dest_aligned_order = list(dest_aligned.get_current_order())

    print(f"  Aligned dest order: {dest_aligned_order[:10]}..." if len(dest_aligned_order) > 10 else f"  Aligned dest order: {dest_aligned_order}")

    result2 = builder.build([source.deep_copy(), dest_aligned.deep_copy()])

    final2 = list(result2.interpolated_trees[-1].get_current_order())
    final2_splits = set(result2.interpolated_trees[-1].to_weighted_splits().keys())
    topo_match2 = final2_splits == dest_splits
    order_match2 = final2 == dest_aligned_order

    movers2 = set()
    for subtree in result2.current_subtree_tracking:
        if subtree:
            movers2.update(subtree.taxa)

    print(f"  Movers detected: {sorted(movers2)}")
    print(f"  Total trees: {len(result2.interpolated_trees)}")
    print(f"  Topology match: {topo_match2}")
    print(f"  Order match (vs aligned): {order_match2}")

    # Compare movement
    print("\n--- Movement Comparison ---")
    print(f"  Same movers: {movers1 == movers2}")
    print(f"  Same tree count: {len(result1.interpolated_trees) == len(result2.interpolated_trees)}")

    # Check if non-movers preserved source order
    non_movers = set(source_order) - movers1
    source_non_mover_order = [t for t in source_order if t in non_movers]
    final1_non_mover_order = [t for t in final1 if t in non_movers]
    final2_non_mover_order = [t for t in final2 if t in non_movers]

    print(f"  Non-movers preserved in Test 1: {source_non_mover_order == final1_non_mover_order}")
    print(f"  Non-movers preserved in Test 2: {source_non_mover_order == final2_non_mover_order}")

    return topo_match1, topo_match2


def main():
    jt_logger.disabled = True

    results = []

    # Test 1: reverse_bootstrap_52 - 3 movers of different sizes
    results.append(test_case(
        "reverse_bootstrap_52 (3 movers: sizes 1, 3, 7)",
        "((((((((13,14),17),27),((((15,16),23),((((((18,19),20),21),22),24),25)),((((((26,28),29),30),31),32),33))),((((34,35),36),37),38)),((((((39,40),41),42),43),44),45)),((((46,47),48),49),50)),51),52);",
        "((((((((13,14),((((15,16),23),17),27)),((((((18,19),20),21),22),24),25)),((((((26,28),29),30),31),32),33)),((((34,35),36),37),38)),((((((39,40),41),42),43),44),45)),((((46,47),48),49),50)),51),52);",
        {"27": ["27"], "15_16_23": ["15", "16", "23"], "18_25": ["18", "19", "20", "21", "22", "24", "25"]}
    ))

    # Test 2: heiko_4 - multiple small movers
    results.append(test_case(
        "heiko_4 (movers: E, F, G, D, H, C)",
        "((O1,O2),(((A,B),C),((((F,G),E),D),H)));",
        "((O1,O2),(((A,B),(C,(F,(E,G)))),(D,H)));",
        {"E": ["E"], "F": ["F"], "G": ["G"]}
    ))

    # Test 3: bird_trees - single mover but large clades
    results.append(test_case(
        "bird_trees (1 mover: Ostrich)",
        "(Emu,(((((((BrushTurkey,Chicken),(magpiegoose,duck)),((LBPenguin,GaviaStellata),(oystercatcher,turnstone))),(GreatRhea,LesserRhea)),Ostrich),(((lbmoa,EasternMoa),Dinornis),((Alligator,Caiman),(ECtinamou,(Gtinamou,Crypturellus))))),(BrownKiwi,(LSKiwi,gskiwi))),Cassowary);",
        "(Emu,((Ostrich,((((lbmoa,EasternMoa),Dinornis),((Alligator,Caiman),(ECtinamou,(Gtinamou,Crypturellus)))),((((BrushTurkey,Chicken),(magpiegoose,duck)),((LBPenguin,GaviaStellata),(oystercatcher,turnstone))),(GreatRhea,LesserRhea)))),(BrownKiwi,(LSKiwi,gskiwi))),Cassowary);",
        {"Ostrich": ["Ostrich"]}
    ))

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for i, (topo1, topo2) in enumerate(results):
        print(f"  Test {i+1}: Topo match (no align)={topo1}, Topo match (aligned)={topo2}")


if __name__ == "__main__":
    main()
