"""
Debug script to verify the "tabula rasa" (clean slate) approach.

The first subtree should collapse ALL splits from the source tree,
creating a blank slate before rebuilding with expand operations.

This tests the complete_interpolation_workflow test case.
"""

from brancharchitect.parser import parse_newick
from brancharchitect.tree_interpolation.subtree_paths.planning.builder import (
    build_edge_plan,
)
from brancharchitect.tree_interpolation.consensus_tree.intermediate_tree import (
    calculate_intermediate_tree,
)
from brancharchitect.consensus.consensus_tree import apply_split_in_tree
from brancharchitect.jumping_taxa.lattice.iterate_lattice_algorithm import (
    iterate_lattice_algorithm,
)
from brancharchitect.elements.partition_set import PartitionSet


def prepare_simple_subtree_paths(tree1, tree2, active_edge, jumping_subtrees):
    """
    Simple test helper to prepare subtree paths based on overlap.

    This assigns splits to subtrees based on whether they overlap with the subtree indices.
    Note: This does NOT account for incompatible splits added by collapse-first strategy.
    """
    to_collapse = tree1.to_splits() - tree2.to_splits()
    to_expand = tree2.to_splits() - tree1.to_splits()

    collapse_splits_by_subtree = {}
    expand_splits_by_subtree = {}

    for subtree in jumping_subtrees[active_edge]:
        # Assign collapse splits that overlap with this subtree
        if subtree not in collapse_splits_by_subtree:
            collapse_splits_by_subtree[subtree] = PartitionSet(
                encoding=tree1.taxa_encoding
            )

        for split in to_collapse:
            # If split overlaps with subtree, assign it
            if set(subtree.indices) & set(split.indices):
                collapse_splits_by_subtree[subtree].add(split)

        # Assign expand splits that overlap with this subtree
        if subtree not in expand_splits_by_subtree:
            expand_splits_by_subtree[subtree] = PartitionSet(
                encoding=tree1.taxa_encoding
            )

        for split in to_expand:
            # If split overlaps with subtree, assign it
            if set(subtree.indices) & set(split.indices):
                expand_splits_by_subtree[subtree].add(split)

    return {
        "collapse_splits_by_subtree": collapse_splits_by_subtree,
        "expand_splits_by_subtree": expand_splits_by_subtree,
        "all_collapse_splits": to_collapse,
        "all_expand_splits": to_expand,
    }


def main():
    # Test case from TestCompleteSplitHandling.test_complete_interpolation_workflow
    tree1_newick = "(((A:1,(A1:1,A2:1):1):1,(B:1,B1:1):1):1,((C:1,(D:1,E:1):1):1,((F:1,I:1):1,(G:1,M:1):1):1):1,(H:1,(O1:1,O2:1):1):1);"
    tree2_newick = "(((A:1,A1:1):1,(B1:1,A2:1):1):1,((C:1,(D:1,E:1):1):1,((F:1,M:1):1,(G:1,I:1):1):1):1,(H:1,(O1:1,O2:1):1):1);"

    # Define taxa order to ensure consistent encoding
    taxa_order = [
        "O1",
        "O2",
        "A",
        "A1",
        "A2",
        "B",
        "B1",
        "C",
        "D",
        "E",
        "F",
        "G",
        "I",
        "M",
        "H",
    ]

    print("=" * 80)
    print("TABULA RASA DEBUG SCRIPT")
    print("=" * 80)
    print()

    # Parse trees with shared taxa order
    print("Parsing trees with shared encoding...")
    tree1 = parse_newick(tree1_newick, order=taxa_order)
    tree2 = parse_newick(tree2_newick, order=taxa_order)
    print(f"Tree 1 has {len(tree1.to_splits())} splits")
    print(f"Tree 2 has {len(tree2.to_splits())} splits")
    print()

    # Get all splits from both trees
    tree1_splits = tree1.to_splits()
    tree2_splits = tree2.to_splits()

    print("Tree 1 splits:")
    for split in sorted(tree1_splits, key=lambda s: len(s.indices)):
        print(f"  {list(split.indices)}")
    print()

    print("Tree 2 splits:")
    for split in sorted(tree2_splits, key=lambda s: len(s.indices)):
        print(f"  {list(split.indices)}")
    print()

    # Find jumping subtrees
    print("Running lattice algorithm to find jumping subtrees...")
    jumping_subtrees, _ = iterate_lattice_algorithm(
        tree1, tree2, list(tree1.taxa_encoding.keys())
    )

    if not jumping_subtrees:
        print("No jumping subtrees found - trees are compatible")
        return

    print(f"Found {len(jumping_subtrees)} changing edge(s)")
    for edge, partitions in jumping_subtrees.items():
        print(f"  Edge {list(edge.indices)}: {len(partitions)} partition(s)")
    print()

    # Get first changing edge and first solution
    active_edge = next(iter(jumping_subtrees.keys()))
    partitions = jumping_subtrees[active_edge]

    # Convert to dict format expected by helper
    jumping_dict = {active_edge: partitions}

    print(f"Processing edge: {list(active_edge.indices)}")
    print(f"Using {len(partitions)} subtree partitions")
    print()

    # Prepare subtree paths
    print("=" * 80)
    print("STEP 1: PREPARE SUBTREE PATHS")
    print("=" * 80)
    subtree_paths = prepare_simple_subtree_paths(
        tree1, tree2, active_edge, jumping_dict
    )

    print(f"Number of subtrees: {len(subtree_paths['collapse_splits_by_subtree'])}")
    print()

    for subtree, collapse_splits in subtree_paths["collapse_splits_by_subtree"].items():
        expand_splits = subtree_paths["expand_splits_by_subtree"][subtree]
        print(f"Subtree {list(subtree.indices)}:")
        print(f"  Assigned collapse splits: {len(collapse_splits)}")
        for split in sorted(collapse_splits, key=lambda s: len(s.indices)):
            print(f"    {list(split.indices)}")
        print(f"  Assigned expand splits: {len(expand_splits)}")
        for split in sorted(expand_splits, key=lambda s: len(s.indices)):
            print(f"    {list(split.indices)}")
        print()

    # Get all collapse/expand splits
    all_collapse = subtree_paths["all_collapse_splits"]
    all_expand = subtree_paths["all_expand_splits"]

    print(f"Total collapse splits (from tree1): {len(all_collapse)}")
    print(f"Total expand splits (to tree2): {len(all_expand)}")
    print()

    # Build edge plan
    print("=" * 80)
    print("STEP 2: BUILD EDGE PLAN")
    print("=" * 80)
    plan = build_edge_plan(
        subtree_paths["expand_splits_by_subtree"],
        subtree_paths["collapse_splits_by_subtree"],
        tree1,
        tree2,
        active_edge,
    )

    print(f"Plan generated for {len(plan)} subtree(s)")
    print()

    subtree_number = 1
    for subtree, subtree_plan in plan.items():
        is_first = subtree_number == 1
        print(
            f"{'[FIRST SUBTREE]' if is_first else '[SUBTREE]'} {list(subtree.indices)}:"
        )

        collapse_path = subtree_plan["collapse"]["path_segment"]
        expand_path = subtree_plan["expand"]["path_segment"]

        print(f"  Collapse path: {len(collapse_path)} splits")
        for split in sorted(collapse_path, key=lambda s: len(s.indices)):
            print(f"    {list(split.indices)}")

        print(f"  Expand path: {len(expand_path)} splits")
        for split in sorted(expand_path, key=lambda s: len(s.indices)):
            print(f"    {list(split.indices)}")
        print()

        subtree_number += 1

    # Analyze: Does first subtree collapse ALL tree1 splits?
    print("=" * 80)
    print("ANALYSIS: TABULA RASA CHECK")
    print("=" * 80)

    first_subtree = next(iter(plan.keys()))
    first_collapse = plan[first_subtree]["collapse"]["path_segment"]

    print(f"Tree 1 total splits: {len(tree1_splits)}")
    print(f"First subtree collapse splits: {len(first_collapse)}")
    print()

    # Find what's missing
    missing_from_first_collapse = tree1_splits - first_collapse

    if missing_from_first_collapse:
        print(
            f"❌ PROBLEM: {len(missing_from_first_collapse)} splits NOT collapsed by first subtree:"
        )
        for split in sorted(missing_from_first_collapse, key=lambda s: len(s.indices)):
            print(f"  {list(split.indices)}")
        print()
        print("These splits will remain in the tree and may cause incompatibility!")
    else:
        print(
            "✅ SUCCESS: First subtree collapses ALL tree1 splits (tabula rasa achieved)"
        )
    print()

    # Execute plan to verify
    print("=" * 80)
    print("STEP 3: EXECUTE PLAN")
    print("=" * 80)

    current_tree = tree1.deep_copy()
    print(f"Starting tree splits: {len(current_tree.to_splits())}")
    print()

    for subtree, subtree_plan in plan.items():
        print(f"Processing subtree {list(subtree.indices)}...")

        # Collapse phase
        collapse_splits = subtree_plan["collapse"]["path_segment"]
        if collapse_splits:
            print(f"  Collapsing {len(collapse_splits)} splits...")
            for s in sorted(collapse_splits, key=lambda x: len(x.indices)):
                print(f"    Setting length=0: {list(s.indices)}")
            split_dict = {s: 0.0 for s in collapse_splits}

            # Check if splits are in tree
            current_splits_before = current_tree.to_splits()
            for s in collapse_splits:
                if s not in current_splits_before:
                    print(
                        f"    ❌ WARNING: Split {list(s.indices)} NOT in current tree!"
                    )

            try:
                before_splits = len(current_splits_before)
                current_tree = calculate_intermediate_tree(current_tree, split_dict)
                after_intermediate = len(current_tree.to_splits())
                print(
                    f"    After calculate_intermediate_tree: {after_intermediate} splits (was {before_splits})"
                )

                # IMPORTANT: Must collapse zero-length branches!
                from brancharchitect.tree_interpolation.consensus_tree.consensus_tree import (
                    collapse_zero_length_branches_for_node,
                )

                active_node = current_tree.find_node_by_split(active_edge)
                if active_node:
                    print(
                        f"    Calling collapse_zero_length_branches_for_node on edge {list(active_edge.indices)}"
                    )
                    collapse_zero_length_branches_for_node(active_node)
                    # Refresh splits after collapsing
                    current_tree.invalidate_caches()
                    current_tree.initialize_split_indices(current_tree.taxa_encoding)
                else:
                    print(
                        f"    ❌ WARNING: Could not find active_node for edge {list(active_edge.indices)}"
                    )

                remaining_splits = len(current_tree.to_splits())
                print(f"  After collapse: {remaining_splits} splits remaining")

                # Check if the collapse splits are still there
                current_splits_set = current_tree.to_splits()
                still_there = [s for s in collapse_splits if s in current_splits_set]
                if still_there:
                    print(
                        f"    ❌ WARNING: {len(still_there)} collapse splits STILL in tree:"
                    )
                    for s in still_there:
                        print(f"      {list(s.indices)}")

            except Exception as e:
                print(f"  ❌ ERROR during collapse: {e}")
                import traceback

                traceback.print_exc()
                return

        # Expand phase
        expand_splits = subtree_plan["expand"]["path_segment"]
        if expand_splits:
            print(f"  Expanding {len(expand_splits)} splits...")

            for expand_split in expand_splits:
                if expand_split not in current_tree.to_splits():
                    try:
                        apply_split_in_tree(expand_split, current_tree)
                    except Exception as e:
                        print(
                            f"  ❌ ERROR applying expand split {list(expand_split.indices)}: {e}"
                        )

                        # Show current tree state
                        current_splits = current_tree.to_splits()
                        print(f"\n  Current tree has {len(current_splits)} splits:")
                        for s in sorted(current_splits, key=lambda x: len(x.indices))[
                            :15
                        ]:
                            print(f"    {list(s.indices)}")

                        return

            remaining_splits = len(current_tree.to_splits())
            print(f"  After expand: {remaining_splits} splits")

        print()

    # Final check
    print("=" * 80)
    print("FINAL VERIFICATION")
    print("=" * 80)

    final_splits = current_tree.to_splits()
    target_splits = tree2.to_splits()

    print(f"Final tree splits: {len(final_splits)}")
    print(f"Target tree splits: {len(target_splits)}")

    if final_splits == target_splits:
        print("✅ SUCCESS: Final tree matches target tree perfectly!")
    else:
        missing = target_splits - final_splits
        extra = final_splits - target_splits

        if missing:
            print(f"❌ Missing {len(missing)} splits:")
            for s in sorted(missing, key=lambda x: len(x.indices)):
                print(f"  {list(s.indices)}")

        if extra:
            print(f"❌ Extra {len(extra)} splits:")
            for s in sorted(extra, key=lambda x: len(x.indices)):
                print(f"  {list(s.indices)}")
    print()


if __name__ == "__main__":
    main()
