#!/usr/bin/env python
"""
Detailed debug of the reordering issue for Ostrich movement.
"""

from brancharchitect.io import read_newick
from brancharchitect.tree import Node
from brancharchitect.elements.partition import Partition


def debug_reorder_taxa(tree: Node, new_order: list, label: str):
    """Debug the reorder_taxa method."""
    print(f"\n--- {label} ---")
    print(f"Current order: {list(tree.get_current_order())}")
    print(f"Target order:  {new_order}")

    # Check if orders match
    current = list(tree.get_current_order())
    if current == new_order:
        print("Orders already match - no change needed")
        return

    # Apply reordering
    try:
        tree.reorder_taxa(new_order)
        result = list(tree.get_current_order())
        print(f"Result order:  {result}")

        if result == new_order:
            print("SUCCESS: Reordering achieved target order")
        else:
            print("FAILURE: Reordering did NOT achieve target order")
            # Find differences
            for i, (r, t) in enumerate(zip(result, new_order)):
                if r != t:
                    print(f"  Position {i}: got '{r}', expected '{t}'")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()


def main():
    # Load trees
    trees = read_newick("small_example copy 3.tree", treat_zero_as_epsilon=True)

    source = trees[3].deep_copy()
    dest = trees[4].deep_copy()

    print("=" * 80)
    print("ANALYZING OSTRICH MOVEMENT")
    print("=" * 80)

    # Get the pivot edge (the large clade)
    # From the debug output, we know the pivot edge contains these taxa
    pivot_taxa = frozenset(
        {
            "Caiman",
            "oystercatcher",
            "turnstone",
            "lbmoa",
            "ECtinamou",
            "EasternMoa",
            "Gtinamou",
            "LesserRhea",
            "duck",
            "GreatRhea",
            "Alligator",
            "BrushTurkey",
            "GaviaStellata",
            "Ostrich",
            "LBPenguin",
            "Dinornis",
            "Chicken",
            "Crypturellus",
            "magpiegoose",
        }
    )

    # Find the pivot edge partition
    encoding = source.taxa_encoding
    pivot_indices = tuple(sorted(encoding[t] for t in pivot_taxa))
    pivot_edge = Partition(pivot_indices, encoding)

    print(f"\nPivot edge taxa: {sorted(pivot_taxa)}")

    # Find subtrees
    source_subtree = source.find_node_by_split(pivot_edge)
    dest_subtree = dest.find_node_by_split(pivot_edge)

    if source_subtree is None:
        print("ERROR: Pivot edge not found in source!")
        return
    if dest_subtree is None:
        print("ERROR: Pivot edge not found in dest!")
        return

    source_order = list(source_subtree.get_current_order())
    dest_order = list(dest_subtree.get_current_order())

    print(f"\nSource subtree order: {source_order}")
    print(f"Dest subtree order:   {dest_order}")

    # The target order for Ostrich movement should place Ostrich after LBPenguin
    # Let's manually compute what the order should be

    # Ostrich is the mover
    mover_leaves = {"Ostrich"}

    # Source anchors (non-movers in source order)
    source_anchors = [t for t in source_order if t not in mover_leaves]
    print(f"\nSource anchors: {source_anchors}")

    # Anchors before Ostrich in destination
    anchors_before_dest = []
    for t in dest_order:
        if t in mover_leaves:
            break
        anchors_before_dest.append(t)
    print(f"Anchors before Ostrich in dest: {anchors_before_dest}")

    # The correct new order should be:
    # anchors up to insertion point + Ostrich + remaining anchors
    anchor_rank_src = {a: i for i, a in enumerate(source_anchors)}
    anchors_before_in_src_order = sorted(
        (a for a in anchors_before_dest if a in anchor_rank_src),
        key=lambda a: anchor_rank_src[a],
    )
    insertion_index = len(anchors_before_in_src_order)

    print(f"Anchors before (in src order): {anchors_before_in_src_order}")
    print(f"Insertion index: {insertion_index}")

    # Build new order
    new_order = list(source_anchors)
    new_order.insert(insertion_index, "Ostrich")
    print(f"\nExpected new subtree order: {new_order}")

    # Now test reorder_taxa on the subtree
    print("\n" + "=" * 80)
    print("TESTING reorder_taxa ON SUBTREE")
    print("=" * 80)

    test_subtree = source_subtree.deep_copy()
    debug_reorder_taxa(test_subtree, new_order, "Subtree reorder")

    # Also test on full tree
    print("\n" + "=" * 80)
    print("TESTING reorder_taxa ON FULL TREE")
    print("=" * 80)

    # Build full tree order with Ostrich moved
    full_source_order = list(source.get_current_order())
    print(f"Full source order: {full_source_order}")

    # Find where the subtree taxa are in the full order
    subtree_taxa_set = set(source_order)
    subtree_start = None
    for i, t in enumerate(full_source_order):
        if t in subtree_taxa_set:
            subtree_start = i
            break

    print(f"Subtree starts at position {subtree_start} in full tree")

    # Build full target order
    full_target_order = []
    subtree_idx = 0
    for t in full_source_order:
        if t in subtree_taxa_set:
            # Replace with new subtree order
            if subtree_idx < len(new_order):
                full_target_order.append(new_order[subtree_idx])
                subtree_idx += 1
        else:
            full_target_order.append(t)

    print(f"Full target order: {full_target_order}")

    test_full = source.deep_copy()
    debug_reorder_taxa(test_full, full_target_order, "Full tree reorder")

    # Check the actual reordering function
    print("\n" + "=" * 80)
    print("TESTING reorder_tree_toward_destination FUNCTION")
    print("=" * 80)

    from brancharchitect.tree_interpolation.subtree_paths.execution.reordering import (
        reorder_tree_toward_destination,
    )

    # Create the moving subtree partition for Ostrich
    ostrich_idx = encoding["Ostrich"]
    moving_partition = Partition((ostrich_idx,), encoding)

    print(f"Moving partition: {moving_partition.taxa}")

    result = reorder_tree_toward_destination(
        source_tree=source.deep_copy(),
        destination_tree=dest,
        current_pivot_edge=pivot_edge,
        moving_subtree_partition=moving_partition,
    )

    result_order = list(result.get_current_order())
    print(f"\nResult order: {result_order}")

    # Check Ostrich position
    ostrich_pos = result_order.index("Ostrich")
    print(f"Ostrich position in result: {ostrich_pos}")

    dest_full_order = list(dest.get_current_order())
    ostrich_dest_pos = dest_full_order.index("Ostrich")
    print(f"Ostrich position in dest: {ostrich_dest_pos}")

    if ostrich_pos == ostrich_dest_pos:
        print("SUCCESS: Ostrich moved to correct position!")
    else:
        print(f"FAILURE: Ostrich at {ostrich_pos}, should be at {ostrich_dest_pos}")


if __name__ == "__main__":
    main()
