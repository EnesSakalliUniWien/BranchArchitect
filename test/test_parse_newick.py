from brancharchitect.newick_parser import parse_newick
from brancharchitect.deletion_algorithm import get_child

def test_parse_newick_1():
    s = "(,,(,));"
    root = parse_newick(s)
    assert len(root.children) == 3


def test_parse_newick_2():
    s = "(A,B,(C,D));"
    root = parse_newick(s)
    print(root)
    assert len(root.children) == 3
    assert len(root.children[2].children) == 2
    assert get_child(root, 0).name == "A"
    assert get_child(root, 1).name == "B"
    assert get_child(root, 2, 0).name == "C"
    assert get_child(root, 2, 1).name == "D"


def test_parse_newick_3():
    s = "(A,B,(C,D)E)F;"
    root = parse_newick(s)
    assert get_child(root, 0).name == "A"
    assert get_child(root, 1).name == "B"
    assert get_child(root, 2, 0).name == "C"
    assert get_child(root, 2, 1).name == "D"
    assert get_child(root, 2).name == "E"
    assert get_child(root).name == "F"


def test_parse_newick_4():
    s = "(:0.1,:0.2,(:0.3,:0.4):0.5);"
    root = parse_newick(s)
    assert get_child(root, 0).length == 0.1
    assert get_child(root, 1).length == 0.2
    assert get_child(root, 2, 0).length == 0.3
    assert get_child(root, 2, 1).length == 0.4
    assert get_child(root, 2).length == 0.5


def test_parse_newick_5():
    s = "(:0.1,:0.2,(:0.3,:0.4):0.5):0.0;"
    root = parse_newick(s)
    assert get_child(root, 0).length == 0.1
    assert get_child(root, 1).length == 0.2
    assert get_child(root, 2, 0).length == 0.3
    assert get_child(root, 2, 1).length == 0.4
    assert get_child(root, 2).length == 0.5
    assert get_child(root).length == 0.0


def test_parse_newick_6():
    s = "(A:0.1,B:0.2,(C:0.3,D:0.4):0.5);"
    root = parse_newick(s)

    assert get_child(root, 0).length == 0.1
    assert get_child(root, 1).length == 0.2
    assert get_child(root, 2, 0).length == 0.3
    assert get_child(root, 2, 1).length == 0.4
    assert get_child(root, 2).length == 0.5

    assert get_child(root, 0).name == "A"
    assert get_child(root, 1).name == "B"
    assert get_child(root, 2, 0).name == "C"
    assert get_child(root, 2, 1).name == "D"


def test_parse_newick_7():
    s = "(A:0.1,B:0.2,(C:0.3,D:0.4)E:0.5)F;"
    root = parse_newick(s)

    assert get_child(root, 0).length == 0.1
    assert get_child(root, 1).length == 0.2
    assert get_child(root, 2, 0).length == 0.3
    assert get_child(root, 2, 1).length == 0.4
    assert get_child(root, 2).length == 0.5

    assert get_child(root, 0).name == "A"
    assert get_child(root, 1).name == "B"
    assert get_child(root, 2, 0).name == "C"
    assert get_child(root, 2, 1).name == "D"
    assert get_child(root, 2).name == "E"
    assert get_child(root).name == "F"


def test_parse_newick_8():
    s = "((B:0.2,(C:0.3,D:0.4)E:0.5)F:0.1)A;"
    root = parse_newick(s)

    assert get_child(root).name == "A"
    assert get_child(root, 0).name == "F"
    assert get_child(root, 0, 0).name == "B"
    assert get_child(root, 0, 1).name == "E"
    assert get_child(root, 0, 1, 0).name == "C"
    assert get_child(root, 0, 1, 1).name == "D"

    assert get_child(root, 0).length == 0.1
    assert get_child(root, 0, 0).length == 0.2
    assert get_child(root, 0, 1).length == 0.5
    assert get_child(root, 0, 1, 0).length == 0.3
    assert get_child(root, 0, 1, 1).length == 0.4


def test_parse_newick_9():
    s = "((B:0.2,(C:0.3,D:0.4)E:0.5)F:0.1)A:0.9;"
    root = parse_newick(s)

    assert get_child(root).name == "A"
    assert get_child(root, 0).name == "F"
    assert get_child(root, 0, 0).name == "B"
    assert get_child(root, 0, 1).name == "E"
    assert get_child(root, 0, 1, 0).name == "C"
    assert get_child(root, 0, 1, 1).name == "D"

    assert get_child(root, 0).length == 0.1
    assert get_child(root, 0, 0).length == 0.2
    assert get_child(root, 0, 1).length == 0.5
    assert get_child(root, 0, 1, 0).length == 0.3
    assert get_child(root, 0, 1, 1).length == 0.4
    assert get_child(root).length == 0.9


def test_parse_newick_10():
    s = "((A,(B,C),((D,E),((F,G),H))),I);"
    root = parse_newick(s)

    assert get_child(root, 0, 0).name == "A"
    assert get_child(root, 0, 1, 0).name == "B"
    assert get_child(root, 0, 1, 1).name == "C"
    assert get_child(root, 0, 2, 0, 0).name == "D"
    assert get_child(root, 0, 2, 0, 1).name == "E"
    assert get_child(root, 0, 2, 1, 0, 0).name == "F"
    assert get_child(root, 0, 2, 1, 0, 1).name == "G"
    assert get_child(root, 0, 2, 1, 1).name == "H"
    assert get_child(root, 1).name == "I"

def test_parse_newick_11_metadata():
    s = "((A[value=3],(B,C),((D,E),((F,G),H))),I);"
    root = parse_newick(s)

    assert get_child(root, 0, 0).values['value'] == 3
