from brancharchitect.newick_parser import parse_newick
from brancharchitect.jumping_taxa import call_jumping_taxa

def test_jumping_taxa_1():
    s1 = '(A,((D,(E,F)),(G,H)));'
    s2 = '(A,((D,F),((G,E),H)));'

    #s1 = "(((A:1,B:1):1,(C:1,D:1):1):1,(O1:1,O2:1):1);",
    #s2 = "(((A:1,B:1,D:1):1,C:1):1,(O1:1,O2:1):1);",


    t1 = parse_newick(s1)
    t2 = parse_newick(s2)

    jt = call_jumping_taxa(t1, t2)

    print(jt)

    assert False


    #["X1", "X2", "X3", "X4"]

