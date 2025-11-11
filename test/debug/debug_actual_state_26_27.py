#!/usr/bin/env python3
"""
Debug actual_state_26_27 - Test failure with 3×2 matrix error.

This script tests the bird phylogeny tree pair that throws:
"Generalized meet product not implemented for 3×2 matrices."

Goal: Determine if this is:
1. A real limitation (3×2 not implemented)
2. A different matrix size encountered during processing
3. An incorrect error message
"""

import json
from pathlib import Path
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.jumping_taxa import call_jumping_taxa


def main():
    """Test the actual_state_26_27 tree pair."""

    # Load test file
    test_file = Path(
        "test/colouring/trees/failing_tree_pair_26_27/actual_state_26_27.json"
    )
    with open(test_file) as f:
        data = json.load(f)

    tree1_newick = data["tree1"]
    tree2_newick = data["tree2"]

    print("=" * 80)
    print("ACTUAL_STATE_26_27 DEBUG")
    print("=" * 80)
    print(f"\nTest: {test_file}")
    print(f"\nError message: {data['comment']}")

    # Parse trees
    print("\n" + "=" * 80)
    print("PARSING TREES")
    print("=" * 80)

    tree1 = parse_newick(tree1_newick)

    # Handle list returns from parse_newick
    if isinstance(tree1, list):
        tree1 = tree1[0]

    # Parse tree2 with same encoding as tree1
    taxa_order = list(tree1.taxa_encoding.keys())
    tree2 = parse_newick(tree2_newick, order=taxa_order, encoding=tree1.taxa_encoding)

    if isinstance(tree2, list):
        tree2 = tree2[0]

    print(f"\nTree 1: {len(tree1.get_leaves())} leaves")
    print(f"Tree 2: {len(tree2.get_leaves())} leaves")

    # Get leaf sets
    leaves1 = set(tree1.get_leaves())
    leaves2 = set(tree2.get_leaves())

    print(f"\nLeaves in common: {len(leaves1 & leaves2)}")
    print(f"Leaves only in tree1: {leaves1 - leaves2}")
    print(f"Leaves only in tree2: {leaves2 - leaves1}")

    # Check splits before running algorithm
    print("\n" + "=" * 80)
    print("SPLIT ANALYSIS")
    print("=" * 80)

    # Compare split_indices (what __eq__ uses) vs to_splits() (what lattice uses)
    print("\n🔍 Comparing split_indices vs to_splits():")
    print(
        f"tree1.split_indices == tree2.split_indices: {tree1.split_indices == tree2.split_indices}"
    )
    print(f"tree1.split_indices: {tree1.split_indices}")
    print(f"tree2.split_indices: {tree2.split_indices}")

    splits1 = tree1.to_splits()
    splits2 = tree2.to_splits()

    print(f"\nTree 1 splits (to_splits): {len(splits1)}")
    print(f"Tree 2 splits (to_splits): {len(splits2)}")

    common_splits = splits1 & splits2
    unique_to_tree1 = splits1 - splits2
    unique_to_tree2 = splits2 - splits1

    print(f"\nCommon splits: {len(common_splits)}")
    print(f"Unique to tree1: {len(unique_to_tree1)}")
    print(f"Unique to tree2: {len(unique_to_tree2)}")

    if unique_to_tree1:
        print("\n📋 Splits unique to tree1:")
        for split in sorted(unique_to_tree1, key=lambda s: len(s.taxa)):
            taxa_list = sorted(split.taxa)
            print(f"  - {taxa_list} (size {len(split.taxa)})")

    if unique_to_tree2:
        print("\n📋 Splits unique to tree2:")
        for split in sorted(unique_to_tree2, key=lambda s: len(s.taxa)):
            taxa_list = sorted(split.taxa)
            print(f"  - {taxa_list} (size {len(split.taxa)})")

    # Check if trees are topologically identical
    print("\n" + "=" * 80)
    print("TOPOLOGY CHECK")
    print("=" * 80)

    if tree1 == tree2:
        print("\n✅ Trees are TOPOLOGICALLY IDENTICAL (tree1 == tree2)")
        print("   (Expected solution [] is CORRECT)")
    else:
        print("\n❌ Trees are TOPOLOGICALLY DIFFERENT (tree1 != tree2)")
        print("   (Algorithm finding jumping taxa is CORRECT)")

    # Also check via split indices
    if tree1.split_indices == tree2.split_indices:
        print("✅ Split indices match (split_indices comparison)")
    else:
        print("❌ Split indices differ (split_indices comparison)")

    # Check symmetric difference
    symmetric_diff = splits1 ^ splits2
    print(f"\n📊 Symmetric difference (unique splits): {len(symmetric_diff)}")
    if len(symmetric_diff) == 0:
        print("✅ No unique splits - trees should be identical")
    else:
        print("❌ Trees have unique splits - they are different")

    # Run jumping taxa algorithm
    print("\n" + "=" * 80)
    print("RUNNING LATTICE ALGORITHM")
    print("=" * 80)

    try:
        result = call_jumping_taxa(tree1, tree2)

        # Convert indices to taxa names
        idx_to_name = {idx: name for name, idx in tree1.taxa_encoding.items()}

        print("\n✅ Algorithm succeeded!")
        print(f"\nSolutions found: {len(result)}")

        solution_names = []
        for partition_tuple in result:
            partition_names = tuple(
                sorted([idx_to_name[idx] for idx in partition_tuple])
            )
            solution_names.append(partition_names)

        for i, solution in enumerate(solution_names, 1):
            print(f"\nSolution {i}: {len(solution)} taxa")
            print(f"  Taxa: {solution}")

        # Compare with expected
        expected = data["solutions"]
        print(f"\nExpected solutions: {expected}")

        if solution_names == expected:
            print("✅ Results match expected!")
        else:
            print("❌ Results differ from expected")
            print(f"\nExpected: {expected}")
            print(f"Got:      {solution_names}")

    except Exception as e:
        print(f"\n❌ Algorithm failed with error:")
        print(f"   {type(e).__name__}: {e}")

        # Try to extract more detail about the matrix size
        import traceback

        print("\nFull traceback:")
        print(traceback.format_exc())

        print("\n" + "=" * 80)
        print("ERROR ANALYSIS")
        print("=" * 80)
        print("\nThis error suggests:")
        print("1. A matrix of unsupported size was encountered")
        print("2. The error message says '3×2' but this should be supported")
        print("3. May need to check if it's actually a different size")
        print("4. Or if there's an issue with matrix classification")


if __name__ == "__main__":
    main()
