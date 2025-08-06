#!/usr/bin/env python3

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.tree import Node
from brancharchitect.elements.partition_set import Partition


def test_tree_properties():
    """Test tree properties and split indices for potential issues."""

    print("=" * 60)
    print("TESTING TREE PROPERTIES AND SPLIT INDICES")
    print("=" * 60)

    # Test 1: Simple tree parsing and split indices
    print("\n1. Testing simple tree parsing and split indices:")
    simple_tree = "(A:0.1,B:0.2)C:0.3;"
    parsed_result = parse_newick(simple_tree)

    # Handle both single tree and list of trees
    if isinstance(parsed_result, list):
        tree = parsed_result[0]
        print(f"Parsed result is a list with {len(parsed_result)} trees")
    else:
        tree = parsed_result
        print(f"Parsed result is a single tree")

    print(f"Tree: {simple_tree}")
    print(f"Root name: '{tree.name}'")
    print(f"Root split_indices: {tree.split_indices}")
    print(f"Root taxa_encoding: {tree.taxa_encoding}")
    print(f"Root children count: {len(tree.children)}")

    for i, child in enumerate(tree.children):
        print(f"  Child {i}: name='{child.name}', split_indices={child.split_indices}")

    # Test 2: Check split indices consistency
    print("\n2. Testing split indices consistency:")
    all_nodes = tree.traverse()
    print(f"Total nodes: {len(all_nodes)}")

    for i, node in enumerate(all_nodes):
        print(
            f"  Node {i}: name='{node.name}', is_leaf={node.is_leaf()}, split_indices={node.split_indices}"
        )

        # Check if split indices are consistent with taxa_encoding
        if node.is_leaf() and node.name:
            expected_index = node.taxa_encoding.get(node.name)
            actual_indices = tuple(node.split_indices)
            if expected_index is not None:
                expected_tuple = (expected_index,)
                if actual_indices != expected_tuple:
                    print(
                        f"    ⚠️  WARNING: Leaf node '{node.name}' has inconsistent split indices!"
                    )
                    print(
                        f"       Expected: {expected_tuple}, Actual: {actual_indices}"
                    )

    # Test 3: Test with metadata tree
    print("\n3. Testing tree with metadata:")
    meta_tree = "(A[value=1]:0.1,B[confidence=0.95]:0.2)C[support=85]:0.3;"
    parsed_meta = parse_newick(meta_tree)

    # Handle both single tree and list of trees
    if isinstance(parsed_meta, list):
        tree_meta = parsed_meta[0]
    else:
        tree_meta = parsed_meta

    print(f"Tree: {meta_tree}")
    print(f"Root values: {tree_meta.values}")

    for i, child in enumerate(tree_meta.children):
        print(f"  Child {i}: name='{child.name}', values={child.values}")

    # Test 4: Test tree operations and cache invalidation
    print("\n4. Testing cache invalidation:")
    tree_copy = tree.deep_copy()
    print(f"Original tree has splits cache: {hasattr(tree, '_splits_cache')}")
    print(f"Copy tree has splits cache: {hasattr(tree_copy, '_splits_cache')}")

    # Force splits computation
    splits = tree.to_splits()
    print(f"Tree splits: {splits}")
    print(f"Splits computed successfully: {splits is not None}")

    # Test 5: Test with complex tree from user's example
    print("\n5. Testing with complex phylogenetic tree:")
    complex_tree = "(KR074182.1:0.000002,(((((KR074151.1:0.000003,KR074152.1:0.010122)98.8/99:0.097371))):0.000002);"

    try:
        parsed_complex = parse_newick(complex_tree)

        # Handle both single tree and list of trees
        if isinstance(parsed_complex, list):
            complex_parsed = parsed_complex[0]
        else:
            complex_parsed = parsed_complex

        print(f"✅ Complex tree parsed successfully")
        print(f"Root children: {len(complex_parsed.children)}")
        print(f"Total nodes: {len(complex_parsed.traverse())}")
        print(
            f"Leaf names sample: {[leaf.name for leaf in complex_parsed.get_leaves()[:5]]}"
        )

        # Check for any anomalies in split indices
        all_nodes_complex = complex_parsed.traverse()
        leaf_nodes = [n for n in all_nodes_complex if n.is_leaf()]
        internal_nodes = [n for n in all_nodes_complex if not n.is_leaf()]

        print(f"Leaf nodes: {len(leaf_nodes)}")
        print(f"Internal nodes: {len(internal_nodes)}")

        # Check split indices ranges
        all_indices = set()
        for node in all_nodes_complex:
            all_indices.update(node.split_indices)

        print(
            f"Split indices range: {min(all_indices) if all_indices else 'N/A'} to {max(all_indices) if all_indices else 'N/A'}"
        )
        print(f"Number of unique indices: {len(all_indices)}")
        print(f"Taxa encoding size: {len(complex_parsed.taxa_encoding)}")

        if len(all_indices) != len(complex_parsed.taxa_encoding):
            print("⚠️  WARNING: Mismatch between unique indices and taxa encoding size!")

    except Exception as e:
        print(f"❌ Error parsing complex tree: {e}")

    # Test 6: Test potential issues in initialization
    print("\n6. Testing potential initialization issues:")

    # Check for default parameter mutation issues
    def create_node_with_defaults():
        return Node()  # Using all defaults

    node1 = create_node_with_defaults()
    node2 = create_node_with_defaults()

    # Check if they share mutable defaults
    node1.children.append(Node(name="test1"))
    node1.values["test"] = "value1"

    if len(node2.children) > 0:
        print("❌ CRITICAL: Mutable default sharing detected in children!")
    else:
        print("✅ Children defaults are properly isolated")

    if "test" in node2.values:
        print("❌ CRITICAL: Mutable default sharing detected in values!")
    else:
        print("✅ Values defaults are properly isolated")

    # Test 7: Test split index consistency after operations
    print("\n7. Testing split index consistency after operations:")
    test_tree = "(A:0.1,(B:0.2,C:0.3):0.1):0.0;"
    parsed_ops = parse_newick(test_tree)

    # Handle both single tree and list of trees
    if isinstance(parsed_ops, list):
        tree_ops = parsed_ops[0]
    else:
        tree_ops = parsed_ops

    print(f"Before reordering:")
    for node in tree_ops.traverse():
        print(f"  {node.name or 'Internal'}: {node.split_indices}")

    # Test reordering
    try:
        tree_ops.reorder_taxa(["C", "B", "A"])
        print(f"After reordering ['C', 'B', 'A']:")
        for node in tree_ops.traverse():
            print(f"  {node.name or 'Internal'}: {node.split_indices}")
    except Exception as e:
        print(f"Error during reordering: {e}")

    print("\n" + "=" * 60)
    print("TREE PROPERTY TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_tree_properties()
