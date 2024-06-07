from brancharchitect.newick_parser import parse_newick
from brancharchitect.jumping_taxa.tree_interpolation import interpolate_tree, interpolate_adjacent_tree_pairs
from brancharchitect.jumping_taxa.deletion_algorithm import get_child



def test_interpolate_tree_1():
    s1 = "((A,(B,C),((D,E),((F,G),H))),I);"
    s2 = "((A,B,((D,E,C),((F,G),H))),I);"

    t1 = parse_newick(s1)
    t2 = parse_newick(s2)

    trajectory = interpolate_tree(t1, t2)
    assert len(trajectory) == 4

def test_interpolate_tree_pairs():
    s1 = "((A,(B,C),((D,E),((F,G),H))),I);"
    s2 = "((A,B,((D,E,C),((F,G),H))),I);"
    s3 = "((A,(B,C),((D,E),((F,G),H))),I);"

    t1 = parse_newick(s1)
    t2 = parse_newick(s2, t1._order)
    t3 = parse_newick(s3, t1._order)

    trajectory = interpolate_adjacent_tree_pairs([t1, t2, t3])
    assert len(trajectory) == 11

def test_interpolate_tree_2():
    s1 = "((A:2,(B,C)),D);"
    s2 = "(((A:1,B),C),D);"

    t1 = parse_newick(s1)
    t2 = parse_newick(s2)

    it1, c1, c2, it2 = interpolate_tree(t1, t2)

    assert get_child(it1, 0, 0).length == 1.5
    assert get_child(it1, 0, 1).length == 0

    assert get_child(c1, 0, 1).name == 'B'
    assert get_child(c1, 0, 2).name == 'C'
