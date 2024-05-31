from brancharchitect.newick_parser import parse_newick

def test_deep_tree():
    N = 2000
    l = []
    for i in range(N):
        l.append('(')
    l.append('0 ')
    for i in range(N):
        l.append(f', {i+1})')
    newick = ''.join(l)

    root = parse_newick(newick)

    root.to_dict()


def test_shallow_tree():
    N = 100
    l = []
    for i in range(N):
        l.append('(')
    l.append('0 ')
    for i in range(N):
        l.append(f', {i+1})')
    newick = ''.join(l)

    root = parse_newick(newick)

    root.to_dict()
