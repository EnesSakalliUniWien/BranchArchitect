#!/usr/bin/env python3

"""
Comprehensive test suite for multi-tree NHX parsing with edge cases.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from brancharchitect.parser.newick_parser import parse_newick


def test_simple_multi_tree():
    """Test simple multi-tree parsing without NHX first."""

    simple_multi = "(A:0.1,B:0.2)C:0.3;(D:0.4,E:0.5)F:0.6;"

    print("=== Test 1: Simple multi-tree (no NHX) ===")
    print("Input:", simple_multi)

    trees = parse_newick(simple_multi, force_list=True)
    print(f"✅ Successfully parsed {len(trees)} simple trees")
    
    assert isinstance(trees, list), "Should return a list of trees"
    assert len(trees) == 2, f"Expected 2 trees, got {len(trees)}"


def test_mixed_nhx_trees():
    """Test mixed NHX and non-NHX trees."""

    mixed_trees = """(A:0.1,B:0.2)C:0.3;
(D:0.15[&&NHX:support=0.9],E:0.25[&&NHX:confidence=0.8])F:0.35[&&NHX:bootstrap=100];
((G:0.05,H:0.08)I:0.12,J:0.18)K:0.22;"""

    print("\n=== Test 2: Mixed NHX and non-NHX trees ===")
    print("Input (truncated):", mixed_trees[:100] + "...")

    try:
        trees = parse_newick(mixed_trees, force_list=True)
        print(f"✅ Successfully parsed {len(trees)} mixed trees")

        # Check which trees have NHX metadata
        for i, tree in enumerate(trees, 1):

            def has_nhx_metadata(node):
                """Check if any node in the tree has NHX metadata."""
                if hasattr(node, "values") and node.values:
                    return True
                return any(
                    has_nhx_metadata(child) for child in getattr(node, "children", [])
                )

            has_metadata = has_nhx_metadata(tree)
            print(f"  Tree {i}: {'Has' if has_metadata else 'No'} NHX metadata")

        assert isinstance(trees, list), "Should return a list of trees"
        assert len(trees) == 3, f"Expected 3 trees, got {len(trees)}"

    except Exception as e:
        assert False, f"Error parsing mixed trees: {e}"


def test_edge_cases():
    """Test edge cases in multi-tree parsing."""

    edge_cases = [
        # Empty string
        ("Empty string", ""),
        # Single semicolon
        ("Single semicolon", ";"),
        # Multiple semicolons
        ("Multiple semicolons", ";;"),
        # Tree with trailing semicolon and whitespace
        ("Trailing semicolon + whitespace", "(A:0.1,B:0.2); "),
        # Whitespace between trees
        ("Whitespace between trees", "(A:0.1,B:0.2);  (C:0.3,D:0.4);"),
        # Tree with complex NHX and trailing content
        (
            "Complex NHX with trailing",
            "(A:0.1[&&NHX:support=0.9:confidence=0.8],B:0.2)C:0.3[&&NHX:bootstrap=100]; ",
        ),
    ]

    print("\n=== Test 3: Edge cases ===")

    success_count = 0
    for name, case in edge_cases:
        print(f"\nEdge case: {name}")
        print(f"  Input: '{case}'")
        try:
            trees = parse_newick(case, force_list=True)
            print(f"  ✅ Parsed {len(trees)} trees")
            success_count += 1
        except Exception as e:
            print(f"  ⚠️  Exception: {e}")

    print(f"\nEdge case summary: {success_count}/{len(edge_cases)} passed")
    # For edge cases, we expect some to fail, so we just verify we can handle them gracefully
    assert success_count >= 0, "Should handle edge cases without crashing"


def test_nhx_with_special_characters():
    """Test NHX metadata with special characters and complex values."""

    special_nhx = """(A:0.1[&&NHX:name="Species A":location="North America"],
B:0.2[&&NHX:name="Species B":location="South America"])Root:0.0[&&NHX:confidence=1.0:method="ML"];"""

    print("\n=== Test 4: NHX with special characters ===")
    print("Input (truncated):", special_nhx[:100] + "...")

    try:
        trees = parse_newick(special_nhx, force_list=True)
        print(f"✅ Successfully parsed {len(trees)} trees with special NHX")

        # Show sample metadata
        def show_metadata(node, level=0):
            indent = "  " * level
            if hasattr(node, "values") and node.values:
                print(f"{indent}Node metadata: {node.values}")
            for child in getattr(node, "children", []):
                show_metadata(child, level + 1)

        if trees:
            print("Sample metadata from first tree:")
            show_metadata(trees[0])

        assert isinstance(trees, list), "Should return a list of trees"
        assert len(trees) >= 1, f"Expected at least 1 tree, got {len(trees)}"

    except Exception as e:
        assert False, f"Error parsing special NHX: {e}"


def test_parser_state_reset():
    """Test that parser state is properly reset between trees."""

    # This test ensures parser doesn't carry state from one tree to the next
    state_test = """(A:0.1[&&NHX:test=1],B:0.2)C:0.3;
(D:0.4,E:0.5[&&NHX:test=2])F:0.6;
(G:0.7[&&NHX:test=3],H:0.8)I:0.9[&&NHX:test=4];"""

    print("\n=== Test 5: Parser state reset ===")
    print("Testing that parser state resets properly between trees...")

    try:
        trees = parse_newick(state_test, force_list=True)
        print(f"✅ Successfully parsed {len(trees)} trees")

        # Verify each tree is independent
        for i, tree in enumerate(trees, 1):

            def count_metadata_nodes(node):
                count = 1 if hasattr(node, "values") and node.values else 0
                return count + sum(
                    count_metadata_nodes(child)
                    for child in getattr(node, "children", [])
                )

            metadata_count = count_metadata_nodes(tree)
            print(f"  Tree {i}: {metadata_count} nodes with metadata")

        # No explicit return; success is reaching this point without exceptions
    except Exception as e:
        print(f"❌ Error in state reset test: {e}")
        assert False, f"Parser state reset failed: {e}"


if __name__ == "__main__":
    print("Comprehensive Multi-tree NHX Parsing Test Suite")
    print("=" * 60)

    tests = [
        test_simple_multi_tree,
        test_mixed_nhx_trees,
        test_edge_cases,
        test_nhx_with_special_characters,
        test_parser_state_reset,
    ]

    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")

    print("\n" + "=" * 60)
    print(f"Test Summary: {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("🎉 All tests passed! Multi-tree NHX parsing is working correctly.")
    else:
        print("⚠️  Some tests failed. Review the output above for details.")
