from brancharchitect.newick_parser import parse_newick
from brancharchitect.jumping_taxa import call_jumping_taxa
import pytest

@pytest.mark.timeout(5)
def test_jumping_taxa_1():
    s = ('(A,((D,(E,F)),(G,H)));' + 
         '(A,((D,F),((G,E),H)));')

    t1, t2  = parse_newick(s)

    jt = call_jumping_taxa(t1, t2)

    assert len(jt) == 1
    assert ('E',) in jt

