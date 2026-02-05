from brancharchitect.parser.newick_parser import parse_newick
import pytest


@pytest.mark.skip()
def test_deep_tree():
    N = 2000
    l = []
    for i in range(N):
        l.append("(")
    l.append("0 ")
    for i in range(N):
        l.append(f", {i + 1})")
    newick = "".join(l)

    root = parse_newick(newick)

    root.to_dict()


def test_shallow_tree():
    N = 100
    l = []
    for i in range(N):
        l.append("(")
    l.append("0 ")
    for i in range(N):
        l.append(f", {i + 1})")
    newick = "".join(l)

    root = parse_newick(newick)

    root.to_dict()


@pytest.mark.skip(reason="circular_tree module removed - cairosvg is optional")
def test_visualisation():
    newick = "((A,(B,C),((D,E),((F,G),H))),I);"
    tree = parse_newick(newick)
    # svg = generate_multiple_circular_trees_svg([tree])
    pass
