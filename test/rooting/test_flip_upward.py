from brancharchitect.tree import Node
from brancharchitect.rooting.core_rooting import _flip_upward


def make_simple_tree():
    #   A
    #  / \
    # B   C
    a = Node(name="A", length=None)
    b = Node(name="B", length=1.0)
    c = Node(name="C", length=2.0)
    a.children = [b, c]
    b.parent = a
    c.parent = a
    return a, b, c


def test_flip_upward_at_leaf():
    a, b, c = make_simple_tree()
    # Reroot at leaf B
    new_root = _flip_upward(b)
    # B should be root
    assert new_root is b
    assert b.parent is None
    # B should have one child: A
    assert b.children == [a]
    # A should have one child: C
    assert a.children == [c]
    assert a.parent is b
    assert c.parent is a
    # No cycles
    visited = set()

    def check_no_cycles(node):
        assert id(node) not in visited, f"Cycle detected at {node.name}"
        visited.add(id(node))
        for child in node.children:
            check_no_cycles(child)
        visited.remove(id(node))

    check_no_cycles(b)


def test_flip_upward_at_internal():
    a, b, c = make_simple_tree()
    # Reroot at internal node A (should be a no-op)
    new_root = _flip_upward(a)
    assert new_root is a
    assert a.parent is None
    assert set(a.children) == {b, c}
    assert b.parent is a
    assert c.parent is a
    # No cycles
    visited = set()

    def check_no_cycles(node):
        assert id(node) not in visited, f"Cycle detected at {node.name}"
        visited.add(id(node))
        for child in node.children:
            check_no_cycles(child)
        visited.remove(id(node))

    check_no_cycles(a)


def test_flip_upward_chain():
    # A - B - C (A is root, C is leaf)
    a = Node(name="A", length=None)
    b = Node(name="B", length=1.0)
    c = Node(name="C", length=2.0)
    a.children = [b]
    b.parent = a
    b.children = [c]
    c.parent = b
    # Reroot at C
    new_root = _flip_upward(c)
    assert new_root is c
    assert c.parent is None
    assert c.children == [b]
    assert b.parent is c
    assert b.children == [a]
    assert a.parent is b
    assert a.children == []
    # No cycles
    visited = set()

    def check_no_cycles(node):
        assert id(node) not in visited, f"Cycle detected at {node.name}"
        visited.add(id(node))
        for child in node.children:
            check_no_cycles(child)
        visited.remove(id(node))

    check_no_cycles(c)


# --- Additional Edge Case Tests ---
def test_flip_upward_single_node():
    # Single node tree
    a = Node(name="A", length=None)
    new_root = _flip_upward(a)
    assert new_root is a
    assert a.parent is None
    assert a.children == []
    # No cycles
    visited = set()

    def check_no_cycles(node):
        assert id(node) not in visited, f"Cycle detected at {node.name}"
        visited.add(id(node))
        for child in node.children:
            check_no_cycles(child)
        visited.remove(id(node))

    check_no_cycles(a)


def test_flip_upward_two_node_tree():
    # A - B
    a = Node(name="A", length=None)
    b = Node(name="B", length=1.0)
    a.children = [b]
    b.parent = a
    # Reroot at B
    new_root = _flip_upward(b)
    assert new_root is b
    assert b.parent is None
    assert b.children == [a]
    assert a.parent is b
    assert a.children == []
    # No cycles
    visited = set()

    def check_no_cycles(node):
        assert id(node) not in visited, f"Cycle detected at {node.name}"
        visited.add(id(node))
        for child in node.children:
            check_no_cycles(child)
        visited.remove(id(node))

    check_no_cycles(b)
    # Reroot back at A
    new_root2 = _flip_upward(a)
    assert new_root2 is a
    assert a.parent is None
    assert a.children == [b]
    assert b.parent is a
    assert b.children == []
    visited = set()
    check_no_cycles(a)


def test_flip_upward_chain_middle():
    # A - B - C - D (A is root, D is leaf)
    a = Node(name="A", length=None)
    b = Node(name="B", length=1.0)
    c = Node(name="C", length=2.0)
    d = Node(name="D", length=3.0)
    a.children = [b]
    b.parent = a
    b.children = [c]
    c.parent = b
    c.children = [d]
    d.parent = c
    # Reroot at C (middle node)
    new_root = _flip_upward(c)
    assert new_root is c
    assert c.parent is None
    assert set(c.children) == {b, d}
    # B should now have one child: A
    assert b.parent is c
    assert b.children == [a]
    assert a.parent is b
    assert a.children == []
    # D should be child of C, parent is C
    assert d.parent is c
    assert d.children == []
    # No cycles
    visited = set()

    def check_no_cycles(node):
        assert id(node) not in visited, f"Cycle detected at {node.name}"
        visited.add(id(node))
        for child in node.children:
            check_no_cycles(child)
        visited.remove(id(node))

    check_no_cycles(c)


def test_flip_upward_multichild():
    #      A
    #    / | \
    #   B  C  D
    #         |
    #         E
    a = Node(name="A", length=None)
    b = Node(name="B", length=1.0)
    c = Node(name="C", length=1.0)
    d = Node(name="D", length=1.0)
    e = Node(name="E", length=1.0)
    a.children = [b, c, d]
    b.parent = a
    c.parent = a
    d.parent = a
    d.children = [e]
    e.parent = d
    # Reroot at D (non-root, non-leaf)
    new_root = _flip_upward(d)
    assert new_root is d
    assert d.parent is None
    # D should have A and E as children
    assert set(d.children) == {a, e}
    assert e.parent is d
    assert e.children == []
    # A should have B and C as children (D is no longer a child)
    assert a.parent is d
    assert set(a.children) == {b, c}
    assert b.parent is a
    assert c.parent is a
    # No cycles
    visited = set()

    def check_no_cycles(node):
        assert id(node) not in visited, f"Cycle detected at {node.name}"
        visited.add(id(node))
        for child in node.children:
            check_no_cycles(child)
        visited.remove(id(node))

    check_no_cycles(d)


def test_flip_upward_all_nodes_reachable():
    # Tree:   A
    #        / \
    #       B   C
    #      /   / \
    #     D   E   F
    a = Node(name="A", length=None)
    b = Node(name="B", length=1.0)
    c = Node(name="C", length=1.0)
    d = Node(name="D", length=1.0)
    e = Node(name="E", length=1.0)
    f = Node(name="F", length=1.0)
    a.children = [b, c]
    b.parent = a
    c.parent = a
    b.children = [d]
    d.parent = b
    c.children = [e, f]
    e.parent = c
    f.parent = c
    # Reroot at E
    new_root = _flip_upward(e)
    # Traverse from new root and collect all nodes
    all_nodes = {a, b, c, d, e, f}
    visited = set()

    def collect_nodes(node):
        assert id(node) not in visited, f"Cycle detected at {node.name}"
        visited.add(id(node))
        for child in node.children:
            collect_nodes(child)

    collect_nodes(new_root)
    # All nodes should be reachable
    found_nodes = set()

    def gather(node):
        found_nodes.add(node)
        for child in node.children:
            gather(child)

    gather(new_root)
    assert all_nodes == found_nodes, (
        f"Not all nodes reachable after rerooting: {all_nodes - found_nodes}"
    )
