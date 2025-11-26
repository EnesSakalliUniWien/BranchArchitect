"""
Test interpolation with small_example.newick file
"""

from brancharchitect.io import read_newick
from brancharchitect.tree_interpolation.sequential_interpolation import (
    SequentialInterpolationBuilder,
)
from brancharchitect.jumping_taxa.lattice.compute_pivot_solutions_with_deletions import (
    compute_pivot_solutions_with_deletions,
)


def test_small_example():
    """Test with the small example newick file"""

    print("=" * 80)
    print("Testing with small_example.newick")
    print("=" * 80)

    # Read trees from file
    trees = read_newick(
        "/Users/berksakalli/Projects/BranchArchitect/current_testfiles/small_example.newick"
    )

    print(f"\nNumber of trees in file: {len(trees)}")

    # Print each tree
    for i, tree in enumerate(trees):
        print(f"\nTree {i + 1}:")
        print(f"  Newick: {tree.to_newick()}")
        print(f"  Number of leaves: {len(tree.get_leaves())}")
        print(f"  Leaf names: {tree.get_current_order()}")

    # Test pairwise interpolation between first two trees
    if len(trees) >= 2:
        print("\n" + "=" * 80)
        print("Testing Pairwise Interpolation (Tree 1 → Tree 2)")
        print("=" * 80)

        tree1 = trees[0]
        tree2 = trees[1]

        # Check s-edges first
        print("\n--- Discovering S-Edges ---")
        jumping_subtree_solutions, deleted_taxa = compute_pivot_solutions_with_deletions(
            tree1, tree2
        )

        print(f"Number of s-edges found: {len(jumping_subtree_solutions)}")
        print(f"Number of iterations with deletions: {len(deleted_taxa)}")

        # Create reverse taxa mapping
        taxa_reverse = {v: k for k, v in tree1.taxa_encoding.items()}

        print("\n--- S-Edge Details ---")
        for i, (s_edge, partitions) in enumerate(
            jumping_subtree_solutions.items(), 1
        ):
            indices = s_edge.resolve_to_indices()
            taxa = [taxa_reverse[idx] for idx in indices]

            print(f"\nS-Edge {i}: {len(taxa)} taxa")
            print(f"  Taxa: {{{', '.join(sorted(taxa))}}}")
            print(f"  Number of partitions: {len(partitions)}")

            if partitions:
                jumping_parts = []
                for partition in partitions:
                    part_indices = partition.resolve_to_indices()
                    part_taxa = [taxa_reverse[idx] for idx in part_indices]
                    jumping_parts.append(f"{{{', '.join(sorted(part_taxa))}}}")
                print(f"    Jumping taxa: {' + '.join(jumping_parts)}")

        print("\n--- Deleted Taxa Per Iteration ---")
        for i, deleted_set in enumerate(deleted_taxa, 1):
            taxa_names = [taxa_reverse[idx] for idx in deleted_set]
            print(f"  Iteration {i}: {{{', '.join(sorted(taxa_names))}}}")

        # Now run full sequential interpolation
        print("\n" + "=" * 80)
        print("Running Full Sequential Interpolation")
        print("=" * 80)

        try:
            sequential_results = SequentialInterpolationBuilder().build([tree1, tree2])

            print(f"\n✓ Success!")
            print(
                f"  Number of interpolated trees: {len(sequential_results.interpolated_trees)}"
            )

            print("\n--- Interpolated Tree Sequence ---")
            for i, tree in enumerate(sequential_results.interpolated_trees):
                newick = tree.to_newick()
                # Print first 100 chars of newick
                if len(newick) > 100:
                    print(f"  Tree {i:2d}: {newick[:100]}...")
                else:
                    print(f"  Tree {i:2d}: {newick}")

            # Verify first and last trees
            print("\n--- Verification ---")
            first_tree_newick = sequential_results.interpolated_trees[0].to_newick()
            last_tree_newick = sequential_results.interpolated_trees[-1].to_newick()

            print(
                f"First interpolated tree matches Tree 1: {first_tree_newick == tree1.to_newick()}"
            )
            print(
                f"Last interpolated tree matches Tree 2: {last_tree_newick == tree2.to_newick()}"
            )

            # Check if trees are valid
            print("\n--- Tree Validity Check ---")
            for i, tree in enumerate(sequential_results.interpolated_trees):
                leaves = tree.get_leaves()
                print(f"  Tree {i:2d}: {len(leaves)} leaves")

        except Exception as e:
            print(f"\n✗ Error during interpolation: {e}")
            import traceback

            traceback.print_exc()

    # Test with all three trees if available
    if len(trees) == 3:
        print("\n" + "=" * 80)
        print("Testing Sequential Interpolation (All 3 Trees)")
        print("=" * 80)

        try:
            sequential_results_all = SequentialInterpolationBuilder().build(trees)

            print(f"\n✓ Success!")
            print(
                f"  Number of interpolated trees: {len(sequential_results_all.interpolated_trees)}"
            )

            # Check transition points
            print("\n--- Tree Transitions ---")
            print(f"  Original trees: {len(trees)}")
            print(
                f"  Interpolated trees: {len(sequential_results_all.interpolated_trees)}"
            )
            print(
                f"  Average trees per transition: {len(sequential_results_all.interpolated_trees) / (len(trees) - 1):.1f}"
            )

        except Exception as e:
            print(f"\n✗ Error during full sequence interpolation: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    test_small_example()
