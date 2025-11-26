"""
Simple verification that the interpolation pipeline works end-to-end
"""

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.tree_interpolation.sequential_interpolation import (
    SequentialInterpolationBuilder,
)

# Test Case: Simple trees
tree1_str = "((((A,B),(C,D)),O));"
tree2_str = "((((A,C),(B,D)),O));"

print("=" * 80)
print("End-to-End Interpolation Test")
print("=" * 80)
print(f"\nTree 1: {tree1_str}")
print(f"Tree 2: {tree2_str}")

trees = parse_newick(tree1_str + tree2_str)

print("\n--- Running interpolation ---")
try:
    sequential_results = SequentialInterpolationBuilder().build(trees)

    print(f"✓ Success!")
    print(
        f"  Number of interpolated trees: {len(sequential_results.interpolated_trees)}"
    )
    print(
        f"  Has interpolation_sequence_labels: {hasattr(sequential_results, 'interpolation_sequence_labels')}"
    )

    if hasattr(sequential_results, "interpolation_sequence_labels"):
        print(f"  Sequence labels: {sequential_results.interpolation_sequence_labels}")

    print("\n--- Interpolated Trees ---")
    for i, tree in enumerate(sequential_results.interpolated_trees):
        newick = tree.to_newick()
        print(f"  Tree {i}: {newick[:80]}{'...' if len(newick) > 80 else ''}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback

    traceback.print_exc()
