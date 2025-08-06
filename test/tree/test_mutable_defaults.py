#!/usr/bin/env python3

from brancharchitect.tree import Node


def test_actual_mutable_default_issues():
    """Test if the mutable default arguments actually cause issues."""

    print("=" * 60)
    print("TESTING ACTUAL MUTABLE DEFAULT BEHAVIOR")
    print("=" * 60)

    print("\n1. Testing if mutable defaults cause shared state:")

    # Create two nodes using defaults
    node1 = Node()
    node2 = Node()

    print(f"node1.children initial: {len(node1.children)}")
    print(f"node2.children initial: {len(node2.children)}")
    print(f"node1.values initial: {node1.values}")
    print(f"node2.values initial: {node2.values}")

    # Check if they share the same object
    print(f"Same children object? {node1.children is node2.children}")
    print(f"Same values object? {node1.values is node2.values}")

    if node1.children is node2.children:
        print("❌ CRITICAL: Children lists are shared between instances!")

        # Demonstrate the problem
        test_child = Node(name="test_child")
        node1.children.append(test_child)

        print(f"After appending to node1.children:")
        print(f"  node1.children length: {len(node1.children)}")
        print(f"  node2.children length: {len(node2.children)}")

        if len(node2.children) > 0:
            print("❌ CONFIRMED: Modifying node1.children affected node2.children!")

    else:
        print("✅ Children lists are properly isolated")

    if node1.values is node2.values:
        print("❌ CRITICAL: Values dicts are shared between instances!")

        # Demonstrate the problem
        node1.values["test"] = "value1"

        print(f"After setting node1.values['test']:")
        print(f"  node1.values: {node1.values}")
        print(f"  node2.values: {node2.values}")

        if "test" in node2.values:
            print("❌ CONFIRMED: Modifying node1.values affected node2.values!")

    else:
        print("✅ Values dicts are properly isolated")

    print("\n2. Testing split_indices sharing:")
    print(f"Same split_indices object? {node1.split_indices is node2.split_indices}")

    if node1.split_indices is node2.split_indices:
        print("❌ CRITICAL: split_indices are shared between instances!")
    else:
        print("✅ split_indices are properly isolated")

    print("\n3. Testing _order sharing:")
    print(f"Same _order object? {node1._order is node2._order}")

    if node1._order is node2._order:
        print("❌ CRITICAL: _order lists are shared between instances!")

        # Demonstrate the problem
        node1._order.append("test_taxon")

        print(f"After appending to node1._order:")
        print(f"  node1._order: {node1._order}")
        print(f"  node2._order: {node2._order}")

        if "test_taxon" in node2._order:
            print("❌ CONFIRMED: Modifying node1._order affected node2._order!")

    else:
        print("✅ _order lists are properly isolated")

    print("\n4. Testing if the issue is mitigated by the 'or []' pattern:")
    print("   The code uses: self.children = children or []")
    print("   This should create new lists even with mutable defaults")

    # The pattern `children or []` should create new empty lists
    # when children=[] (empty list is falsy), so this mitigates the issue partially

    print("\n5. Testing with parse_newick (realistic usage):")
    from brancharchitect.parser.newick_parser import parse_newick

    # Parse two different trees
    tree1 = parse_newick("(A:0.1,B:0.2):0.0;")
    tree2 = parse_newick("(C:0.3,D:0.4):0.0;")

    if isinstance(tree1, list):
        tree1 = tree1[0]
    if isinstance(tree2, list):
        tree2 = tree2[0]

    print(f"tree1 children: {[child.name for child in tree1.children]}")
    print(f"tree2 children: {[child.name for child in tree2.children]}")

    # Check if parsed trees share any objects
    print(f"trees share children list? {tree1.children is tree2.children}")
    print(f"trees share values dict? {tree1.values is tree2.values}")

    if tree1.children is tree2.children or tree1.values is tree2.values:
        print("❌ CRITICAL: Parsed trees share mutable objects!")
    else:
        print("✅ Parsed trees are properly isolated")

    print("\n" + "=" * 60)
    print("MUTABLE DEFAULT TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    test_actual_mutable_default_issues()
