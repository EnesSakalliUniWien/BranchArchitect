from brancharchitect.parser.newick_parser import parse_newick


def test_deep_tree():
    N = 2000
    parts = []
    for i in range(N):
        parts.append("(")
    parts.append("0 ")
    for i in range(N):
        parts.append(f", {i + 1})")
    newick = "".join(parts)

    root = parse_newick(newick)

    root.to_dict()


def test_shallow_tree():
    N = 100
    parts = []
    for i in range(N):
        parts.append("(")
    parts.append("0 ")
    for i in range(N):
        parts.append(f", {i + 1})")
    newick = "".join(parts)

    root = parse_newick(newick)

    root.to_dict()
