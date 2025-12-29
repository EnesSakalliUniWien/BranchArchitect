#!/usr/bin/env python3
"""Debug script to trace the builder flow for pair 5->6"""

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.tree_interpolation.subtree_paths.analysis.split_analysis import (
    get_unique_splits_for_current_pivot_edge_subtree,
    find_incompatible_splits,
)
from brancharchitect.tree_interpolation.subtree_paths.planning.builder import (
    build_edge_plan,
)
from brancharchitect.tree_interpolation.subtree_paths.planning.pivot_split_registry import (
    PivotSplitRegistry,
)


def parse_newick_file(filepath):
    """Parse a newick file with multiple trees."""
    with open(filepath, "r") as f:
        content = f.read()

    tree_strings = [s.strip() + ";" for s in content.split(";") if s.strip()]

    trees = []
    order = None
    for tree_str in tree_strings:
        tree = parse_newick(tree_str, order)
        if order is None:
            order = list(tree.get_current_order())
        trees.append(tree)

    return trees


def main():
    trees = parse_newick_file("52_bootstrap.newick")

    t5 = trees[5]
    t6 = trees[6]

    # Get the root split (pivot edge for the whole tree)
    root_split = Partition(tuple(range(len(t5.taxa_encoding))), t5.taxa_encoding)

    print(f"Root split: {list(root_split.indices)}")

    # Get unique splits for this pivot edge
    collapse_splits, expand_splits = get_unique_splits_for_current_pivot_edge_subtree(
        t5, t6, root_split
    )

    print(f"\nCollapse splits (unique to T5): {len(collapse_splits)}")
    print(f"Expand splits (unique to T6): {len(expand_splits)}")

    # Problem split
    problem_indices = (15, 16, 17, 23, 27)
    problem_split = None
    for s in expand_splits:
        if set(s.indices) == set(problem_indices):
            problem_split = s
            break

    if problem_split:
        print(f"\nProblem split found in expand_splits: {list(problem_split.indices)}")
    else:
        print(f"\nProblem split NOT found in expand_splits!")
        # Check all expand splits
        for s in expand_splits:
            print(f"  - {list(s.indices)}")

    # Check incompatibilities
    all_indices = set(t5.taxa_encoding.values())

    if problem_split:
        incompatible_with_collapse = []
        for s in collapse_splits:
            if not problem_split.is_compatible_with(s, all_indices):
                incompatible_with_collapse.append(s)

        print(f"\nIncompatible collapse splits: {len(incompatible_with_collapse)}")
        for s in incompatible_with_collapse:
            print(f"  - {list(s.indices)}")

    # Now let's simulate what the builder does
    # We need to create subtree assignments
    # For simplicity, let's assume all splits go to the root subtree

    collapse_by_subtree = {root_split: collapse_splits}
    expand_by_subtree = {root_split: expand_splits}

    print("\n=== Simulating builder flow ===")

    # Initialize state
    state = PivotSplitRegistry(
        collapse_splits,
        expand_splits,
        collapse_by_subtree,
        expand_by_subtree,
        root_split,
    )

    print(f"all_collapsible_splits: {len(state.all_collapsible_splits)}")
    print(f"first_subtree_processed: {state.first_subtree_processed}")

    # Get tabula rasa splits
    tabula_rasa = state.get_tabula_rasa_collapse_splits()
    print(f"tabula_rasa collapse splits: {len(tabula_rasa)}")

    # Check if problem split's incompatible splits are in tabula_rasa
    if problem_split:
        missing_incompatible = []
        for s in incompatible_with_collapse:
            if s not in tabula_rasa:
                missing_incompatible.append(s)

        if missing_incompatible:
            print(
                f"\nMISSING incompatible splits from tabula_rasa: {len(missing_incompatible)}"
            )
            for s in missing_incompatible:
                print(f"  - {list(s.indices)}")
        else:
            print(f"\nAll incompatible splits ARE in tabula_rasa - good!")

    # Now let's trace what happens in build_edge_plan
    print("\n=== Calling build_edge_plan ===")
    try:
        plans = build_edge_plan(
            expand_by_subtree,
            collapse_by_subtree,
            t5,
            t6,
            root_split,
        )

        print(f"Plans created for {len(plans)} subtrees")
        for subtree, plan in plans.items():
            collapse_path = plan["collapse"]["path_segment"]
            expand_path = plan["expand"]["path_segment"]
            print(f"\nSubtree: {list(subtree.indices)[:5]}...")
            print(f"  Collapse path: {len(collapse_path)} splits")
            print(f"  Expand path: {len(expand_path)} splits")

            # Check if problem split is in expand path
            if problem_split:
                if problem_split in expand_path:
                    print(f"  Problem split IS in expand path")

                    # Check if all incompatible splits are in collapse path
                    collapse_set = set(collapse_path)
                    missing = [
                        s for s in incompatible_with_collapse if s not in collapse_set
                    ]
                    if missing:
                        print(
                            f"  MISSING incompatible splits from collapse path: {len(missing)}"
                        )
                        for s in missing:
                            print(f"    - {list(s.indices)}")
                    else:
                        print(f"  All incompatible splits ARE in collapse path - good!")
                else:
                    print(f"  Problem split NOT in expand path")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
