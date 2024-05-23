from brancharchitect.parse_utils import parse_newick

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

