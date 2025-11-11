"""
Debug script for simon_test_tree_4 tests.
Analyzes whether algorithm solutions produce topologically identical trees.
"""

import json
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.jumping_taxa import call_jumping_taxa


def detailed_analysis(test_file, test_name):
    """Perform detailed topological analysis of a test case."""
    print("\n" + "=" * 80)
    print(f"DETAILED ANALYSIS: {test_name}")
    print("=" * 80)

    with open(test_file, "r") as f:
        data = json.load(f)

    tree1_str = data["tree1"]
    tree2_str = data["tree2"]
    expected_solutions = data["solutions"]

    print(f"\nTree 1: {tree1_str}")
    print(f"Tree 2: {tree2_str}")
    print(f"Expected solutions: {expected_solutions}")

    # Parse trees
    tree1 = parse_newick(tree1_str)
    if isinstance(tree1, list):
        tree1 = tree1[0]

    tree2 = parse_newick(
        tree2_str, order=list(tree1.taxa_encoding.keys()), encoding=tree1.taxa_encoding
    )
    if isinstance(tree2, list):
        tree2 = tree2[0]

    print(f"\nTaxa ({len(tree1.taxa_encoding)}): {sorted(tree1.taxa_encoding.keys())}")

    # Run algorithm
    print("\n" + "-" * 80)
    print("ALGORITHM EXECUTION")
    print("-" * 80)
    result = call_jumping_taxa(tree1, tree2)

    # Convert to names
    idx_to_name = {idx: name for name, idx in tree1.taxa_encoding.items()}
    solution_names = []
    for partition_tuple in result:
        partition_names = tuple(sorted([idx_to_name[idx] for idx in partition_tuple]))
        solution_names.append(partition_names)
    solution_names = sorted(solution_names)

    print(f"\nAlgorithm found: {solution_names}")
    print(f"Total jumping taxa: {sum(len(p) for p in solution_names)}")

    # Test expected solution
    print("\n" + "-" * 80)
    print("TESTING EXPECTED SOLUTION")
    print("-" * 80)

    for idx, expected_sol in enumerate(expected_solutions):
        print(f"\nExpected solution {idx + 1}: {expected_sol}")
        expected_taxa = [taxon for partition in expected_sol for taxon in partition]
        print(f"Taxa to delete: {expected_taxa}")

        # Convert to indices
        try:
            expected_indices = [tree1.taxa_encoding[name] for name in expected_taxa]
        except KeyError as e:
            print(f"ERROR: Taxon '{e.args[0]}' not found in tree!")
            continue

        # Delete and check
        tree1_exp = tree1.delete_taxa(expected_indices)
        tree2_exp = tree2.delete_taxa(expected_indices)

        remaining = sorted([leaf.name for leaf in tree1_exp.get_leaves()])
        print(f"Remaining taxa ({len(remaining)}): {remaining}")

        newick1_exp = tree1_exp.to_newick()
        newick2_exp = tree2_exp.to_newick()

        print(f"\nTree 1 after: {newick1_exp}")
        print(f"Tree 2 after: {newick2_exp}")

        newick_match = newick1_exp == newick2_exp
        topo_match = tree1_exp == tree2_exp

        print(f"\nNewick identical: {newick_match}")
        print(f"Topologically identical: {topo_match}")

        if not topo_match:
            print("\n⚠️  Expected solution does NOT produce identical trees!")

            # Show splits
            splits1 = set()
            for node in tree1_exp.traverse():
                if not node.is_leaf():
                    leaf_names = frozenset([leaf.name for leaf in node.get_leaves()])
                    if len(leaf_names) > 1 and len(leaf_names) < len(remaining):
                        splits1.add(leaf_names)

            splits2 = set()
            for node in tree2_exp.traverse():
                if not node.is_leaf():
                    leaf_names = frozenset([leaf.name for leaf in node.get_leaves()])
                    if len(leaf_names) > 1 and len(leaf_names) < len(remaining):
                        splits2.add(leaf_names)

            print(
                f"\nTree 1 splits ({len(splits1)}): {sorted([sorted(s) for s in splits1])}"
            )
            print(
                f"Tree 2 splits ({len(splits2)}): {sorted([sorted(s) for s in splits2])}"
            )

            print("\nMissing in Tree 1:")
            for split in sorted([sorted(s) for s in (splits2 - splits1)]):
                print(f"  {split}")
            print("Missing in Tree 2:")
            for split in sorted([sorted(s) for s in (splits1 - splits2)]):
                print(f"  {split}")

    # Test algorithm solution
    print("\n" + "-" * 80)
    print("TESTING ALGORITHM SOLUTION")
    print("-" * 80)

    indices_to_delete = [idx for partition in result for idx in partition]
    deleted_names = sorted([idx_to_name[idx] for idx in indices_to_delete])

    print(f"\nDeleting: {deleted_names}")

    tree1_algo = tree1.delete_taxa(indices_to_delete)
    tree2_algo = tree2.delete_taxa(indices_to_delete)

    remaining_algo = sorted([leaf.name for leaf in tree1_algo.get_leaves()])
    print(f"Remaining taxa ({len(remaining_algo)}): {remaining_algo}")

    newick1_algo = tree1_algo.to_newick()
    newick2_algo = tree2_algo.to_newick()

    print(f"\nTree 1 after: {newick1_algo}")
    print(f"Tree 2 after: {newick2_algo}")

    newick_match_algo = newick1_algo == newick2_algo
    topo_match_algo = tree1_algo == tree2_algo

    print(f"\nNewick identical: {newick_match_algo}")
    print(f"Topologically identical: {topo_match_algo}")

    if topo_match_algo:
        print("\n✓ Algorithm solution produces IDENTICAL trees!")
    else:
        print("\n✗ Algorithm solution does NOT produce identical trees!")

        # Show splits
        splits1 = set()
        for node in tree1_algo.traverse():
            if not node.is_leaf():
                leaf_names = frozenset([leaf.name for leaf in node.get_leaves()])
                if len(leaf_names) > 1 and len(leaf_names) < len(remaining_algo):
                    splits1.add(leaf_names)

        splits2 = set()
        for node in tree2_algo.traverse():
            if not node.is_leaf():
                leaf_names = frozenset([leaf.name for leaf in node.get_leaves()])
                if len(leaf_names) > 1 and len(leaf_names) < len(remaining_algo):
                    splits2.add(leaf_names)

        print(
            f"\nTree 1 splits ({len(splits1)}): {sorted([sorted(s) for s in splits1])}"
        )
        print(f"Tree 2 splits ({len(splits2)}): {sorted([sorted(s) for s in splits2])}")

    # Summary
    print("\n" + "-" * 80)
    print("SUMMARY")
    print("-" * 80)

    return {
        "test_name": test_name,
        "algorithm_solution": solution_names,
        "algorithm_count": sum(len(p) for p in solution_names),
        "algorithm_works": topo_match_algo,
        "expected_works": all(
            tree1.delete_taxa(
                [
                    tree1.taxa_encoding[t]
                    for t in [taxon for partition in sol for taxon in partition]
                ]
            )
            == tree2.delete_taxa(
                [
                    tree2.taxa_encoding[t]
                    for t in [taxon for partition in sol for taxon in partition]
                ]
            )
            for sol in expected_solutions
        ),
    }


# Analyze both tests
result1 = detailed_analysis(
    "test/colouring/trees/simon_test_tree_4/simon_test_tree_4.json",
    "simon_test_tree_4.json",
)

result2 = detailed_analysis(
    "test/colouring/trees/simon_test_tree_4/reverse_simon_test_tree_4.json",
    "reverse_simon_test_tree_4.json",
)

# Final summary
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

for result in [result1, result2]:
    print(f"\n{result['test_name']}:")
    print(f"  Algorithm solution: {result['algorithm_solution']}")
    print(f"  Jumping taxa count: {result['algorithm_count']}")
    print(f"  Algorithm produces identical trees: {result['algorithm_works']}")
    print(f"  Expected solution produces identical trees: {result['expected_works']}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if all(r["algorithm_works"] for r in [result1, result2]):
    print("✓ Algorithm solutions are CORRECT")
    if not all(r["expected_works"] for r in [result1, result2]):
        print("✗ Expected solutions are INCOMPLETE or WRONG")
        print("\nRecommendation: Update test files with algorithm solutions")
    else:
        print("✓ Expected solutions are also correct")
        print("\nNote: Multiple valid solutions may exist")
else:
    print("✗ Algorithm has a BUG - does not produce identical trees")
