#!/usr/bin/env python3

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.tree import Node, ReorderStrategy
from brancharchitect.elements.partition_set import Partition
import traceback


def test_tree_edge_cases():
    """Test edge cases and potential issues in Tree class."""

    print("=" * 60)
    print("TESTING TREE EDGE CASES AND POTENTIAL ISSUES")
    print("=" * 60)

    issues_found = []

    # Test 1: Empty and degenerate trees
    print("\n1. Testing edge cases with empty/degenerate trees:")
    try:
        # Single leaf tree
        single_leaf = parse_newick("A;")
        if isinstance(single_leaf, list):
            single_leaf = single_leaf[0]

        print(
            f"Single leaf tree: '{single_leaf.name}', children: {len(single_leaf.children)}"
        )
        print(f"Split indices: {single_leaf.split_indices}")
        print(f"Taxa encoding: {single_leaf.taxa_encoding}")

        # Check if leaf node has correct split indices
        if single_leaf.is_leaf() and single_leaf.name in single_leaf.taxa_encoding:
            expected_idx = single_leaf.taxa_encoding[single_leaf.name]
            actual_indices = tuple(single_leaf.split_indices)
            if actual_indices != (expected_idx,):
                issues_found.append(
                    f"Single leaf split indices incorrect: expected ({expected_idx},), got {actual_indices}"
                )

    except Exception as e:
        issues_found.append(f"Error with single leaf tree: {e}")

    # Test 2: Taxa encoding consistency
    print("\n2. Testing taxa encoding consistency:")
    try:
        complex_tree_str = "(((A:0.1,B:0.2):0.1,C:0.3):0.1,(D:0.4,E:0.5):0.2):0.0;"
        tree = parse_newick(complex_tree_str)
        if isinstance(tree, list):
            tree = tree[0]

        print(f"Tree: {complex_tree_str}")
        print(f"Taxa encoding: {tree.taxa_encoding}")

        # Check if all leaf names are in taxa encoding
        leaves = tree.get_leaves()
        for leaf in leaves:
            if leaf.name not in tree.taxa_encoding:
                issues_found.append(f"Leaf '{leaf.name}' not found in taxa_encoding")

        # Check if taxa encoding matches current order
        current_order = tree.get_current_order()
        encoding_order = [
            name for name, _ in sorted(tree.taxa_encoding.items(), key=lambda x: x[1])
        ]

        print(f"Current order: {current_order}")
        print(f"Encoding order: {encoding_order}")

        if list(current_order) != encoding_order:
            issues_found.append(
                f"Taxa encoding order mismatch: current={current_order}, encoding={encoding_order}"
            )

    except Exception as e:
        issues_found.append(f"Error with taxa encoding test: {e}")

    # Test 3: Split indices validation
    print("\n3. Testing split indices validation:")
    try:
        tree = parse_newick("((A:0.1,B:0.2):0.1,C:0.3):0.0;")
        if isinstance(tree, list):
            tree = tree[0]

        all_nodes = tree.traverse()
        for node in all_nodes:
            # Check if split indices are within valid range
            for idx in node.split_indices:
                if idx < 0 or idx >= len(tree.taxa_encoding):
                    issues_found.append(
                        f"Split index {idx} out of range for node '{node.name or 'Internal'}'"
                    )

            # Check if internal node split indices are union of children
            if not node.is_leaf() and node.children:
                expected_indices = set()
                for child in node.children:
                    expected_indices.update(child.split_indices)

                actual_indices = set(node.split_indices)
                if actual_indices != expected_indices:
                    issues_found.append(
                        f"Internal node '{node.name or 'Internal'}' split indices don't match children union"
                    )
                    print(
                        f"  Expected: {sorted(expected_indices)}, Got: {sorted(actual_indices)}"
                    )

    except Exception as e:
        issues_found.append(f"Error with split indices validation: {e}")

    # Test 4: Cache invalidation after operations
    print("\n4. Testing cache invalidation:")
    try:
        tree = parse_newick("(A:0.1,(B:0.2,C:0.3):0.1):0.0;")
        if isinstance(tree, list):
            tree = tree[0]

        # Force cache population
        _ = tree.to_splits()

        # Modify tree structure
        if len(tree.children) >= 2 and len(tree.children[1].children) >= 2:
            tree.children[1].swap_children()

            # Check if caches were invalidated
            # This is hard to test directly due to protected members,
            # but we can test that operations still work correctly
            splits_after = tree.to_splits()
            print(f"Splits after modification: {splits_after}")

    except Exception as e:
        issues_found.append(f"Error with cache invalidation test: {e}")

    # Test 5: Deep copy integrity
    print("\n5. Testing deep copy integrity:")
    try:
        original = parse_newick("(A[value=1]:0.1,B[value=2]:0.2)C[value=3]:0.3;")
        if isinstance(original, list):
            original = original[0]

        copy = original.deep_copy()

        # Check if structures are independent
        original.values["test"] = "original"
        copy.values["test"] = "copy"

        if original.values["test"] == copy.values["test"]:
            issues_found.append("Deep copy values not properly isolated")

        # Check if children are independent
        if len(original.children) > 0 and len(copy.children) > 0:
            original.children[0].name = "modified_original"
            if copy.children[0].name == "modified_original":
                issues_found.append("Deep copy children not properly isolated")

        # Check split indices consistency in copy
        for orig_node, copy_node in zip(original.traverse(), copy.traverse()):
            if orig_node.split_indices != copy_node.split_indices:
                issues_found.append(
                    f"Split indices differ between original and copy: {orig_node.split_indices} vs {copy_node.split_indices}"
                )
                break

    except Exception as e:
        issues_found.append(f"Error with deep copy test: {e}")

    # Test 6: Partition constructor edge cases
    print("\n6. Testing Partition constructor edge cases:")
    try:
        tree = parse_newick("(A:0.1,B:0.2):0.0;")
        if isinstance(tree, list):
            tree = tree[0]

        # Test empty partition
        empty_partition = Partition((), tree.taxa_encoding)
        print(f"Empty partition: {empty_partition}")

        # Test partition with duplicate indices (should be handled)
        try:
            dup_partition = Partition((0, 0, 1), tree.taxa_encoding)
            if len(dup_partition) != 2:  # Should deduplicate
                issues_found.append(
                    f"Partition with duplicates not properly handled: {dup_partition}"
                )
        except Exception as e:
            print(f"Partition constructor handles duplicates by raising: {e}")

    except Exception as e:
        issues_found.append(f"Error with Partition constructor test: {e}")

    # Test 7: Metadata handling edge cases
    print("\n7. Testing metadata handling edge cases:")
    try:
        # Test with various metadata types
        meta_tree = parse_newick(
            "(A[int_val=42]:0.1,B[float_val=3.14]:0.2,C[str_val=hello]:0.3);"
        )
        if isinstance(meta_tree, list):
            meta_tree = meta_tree[0]

        leaves = meta_tree.get_leaves()
        for leaf in leaves:
            print(
                f"Leaf '{leaf.name}': values={leaf.values}, types={[type(v).__name__ for v in leaf.values.values()]}"
            )

        # Check if metadata is preserved in deep copy
        copy = meta_tree.deep_copy()
        for orig_leaf, copy_leaf in zip(leaves, copy.get_leaves()):
            if orig_leaf.values != copy_leaf.values:
                issues_found.append(
                    f"Metadata not preserved in deep copy for '{orig_leaf.name}'"
                )

    except Exception as e:
        issues_found.append(f"Error with metadata handling test: {e}")

    # Test 8: Tree manipulation edge cases
    print("\n8. Testing tree manipulation edge cases:")
    try:
        tree = parse_newick("((A:0.1,B:0.2):0.1,(C:0.3,D:0.4):0.2):0.0;")
        if isinstance(tree, list):
            tree = tree[0]

        original_leaf_count = len(tree.get_leaves())
        print(f"Original leaf count: {original_leaf_count}")

        # Test taxa deletion
        try:
            # Delete one taxon (index 0, which should be 'A')
            tree_copy = tree.deep_copy()
            tree_copy.delete_taxa([0])

            new_leaf_count = len(tree_copy.get_leaves())
            print(f"After deleting taxa [0], leaf count: {new_leaf_count}")

            if new_leaf_count != original_leaf_count - 1:
                issues_found.append(
                    f"Taxa deletion didn't reduce leaf count correctly: {original_leaf_count} -> {new_leaf_count}"
                )

            # Check if indices were properly updated
            remaining_leaves = tree_copy.get_leaves()
            all_indices = set()
            for leaf in remaining_leaves:
                all_indices.update(leaf.split_indices)

            if max(all_indices) >= len(tree_copy.taxa_encoding):
                issues_found.append(
                    "Split indices not properly updated after taxa deletion"
                )

        except Exception as e:
            issues_found.append(f"Error with taxa deletion: {e}")

    except Exception as e:
        issues_found.append(f"Error with tree manipulation test: {e}")

    # Test 9: Check for potential memory leaks or circular references
    print("\n9. Testing for potential reference issues:")
    try:
        tree = parse_newick("(A:0.1,B:0.2):0.0;")
        if isinstance(tree, list):
            tree = tree[0]

        # Check parent-child consistency
        for node in tree.traverse():
            for child in node.children:
                if child.parent != node:
                    issues_found.append(
                        f"Parent-child reference inconsistency for node '{child.name or 'Internal'}'"
                    )

        # Check if root has no parent
        if tree.parent is not None:
            issues_found.append("Root node has a parent (should be None)")

    except Exception as e:
        issues_found.append(f"Error with reference consistency test: {e}")

    # Test 10: Performance with large phylogenetic tree names
    print("\n10. Testing with realistic phylogenetic data:")
    try:
        # Use a tree with long taxonomic names similar to user's data
        phylo_tree = "(KR074182_1_Pipistrellus_pipistrellus:0.1,KR074151_1_Pipistrellus_nathusii:0.2)internal_node:0.0;"
        tree = parse_newick(phylo_tree)
        if isinstance(tree, list):
            tree = tree[0]

        print(f"Phylogenetic tree parsed successfully")
        print(f"Taxa encoding: {tree.taxa_encoding}")

        # Check encoding stability after operations
        original_encoding = tree.taxa_encoding.copy()
        _ = tree.to_splits()  # Force split computation

        if tree.taxa_encoding != original_encoding:
            issues_found.append("Taxa encoding changed after split computation")

    except Exception as e:
        issues_found.append(f"Error with phylogenetic data test: {e}")

    # Report results
    print("\n" + "=" * 60)
    print("EDGE CASE TEST RESULTS")
    print("=" * 60)

    if issues_found:
        print(f"❌ Found {len(issues_found)} potential issues:")
        for i, issue in enumerate(issues_found, 1):
            print(f"  {i}. {issue}")
    else:
        print("✅ All edge case tests passed - no issues found!")

    print("=" * 60)


if __name__ == "__main__":
    test_tree_edge_cases()
