from brancharchitect.io import parse_newick
from brancharchitect.rooting.rooting import (
    midpoint_root,
    find_farthest_leaves,
    path_between,
)

############################################################
# HELPER: FOR DIRECT TESTING OF INTERNAL FUNCTIONS
############################################################
# Use the new robust helpers from rooting.py, not from tree.py


def test_midpoint_root_simple():
    # Tree: ((A:2,B:2):2,C:6);
    newick = "((A:2,B:2):2,C:6);"
    tree = parse_newick(newick)
    rooted = midpoint_root(tree)
    children = rooted.children
    lengths = sorted([ch.length for ch in children])
    print("DEBUG: midpoint_root_simple branch lengths:", lengths)
    assert len(children) == 2
    assert all(length >= 0 for length in lengths)


def test_find_farthest_leaves():
    newick = "((A:1,B:1):1,(C:1,D:1):1);"
    tree = parse_newick(newick)
    n1, n2, dist = find_farthest_leaves(tree)
    # All tips are distance 4 apart
    assert dist == 4.0
    assert n1.is_leaf() and n2.is_leaf()
    assert n1.name in {"A", "B", "C", "D"}
    assert n2.name in {"A", "B", "C", "D"}
    assert n1 != n2


def test_path_between():
    newick = "((A:1,B:2):3,C:4);"
    tree = parse_newick(newick)
    leaves = [n for n in tree.traverse() if n.is_leaf()]
    a = [n for n in leaves if n.name == "A"][0]
    c = [n for n in leaves if n.name == "C"][0]
    path = path_between(a, c)
    names = [n.name for n, _ in path]
    assert names[0] == "A"
    assert names[-1] == "C"
    total = sum(edge for _, edge in path[1:])
    assert abs(total - 8.0) < 1e-6
