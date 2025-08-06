from brancharchitect.leaforder.circular_distances import circular_distance, create_ranks
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.consensus.consensus_tree import get_taxa_circular_order


def test_assign_ranks():
    x = tuple(["b", "c", "d", "a"])
    y = tuple(["b", "a", "d", "c"])
    rank_x, rank_y = create_ranks(x, y)
    assert rank_x == tuple([0, 1, 2, 3])
    assert rank_y == tuple([0, 3, 2, 1])


def test_circular_distance():
    x = tuple(["b", "c", "d", "a"])
    y = tuple(["b", "a", "d", "c"])
    distance = circular_distance(x, y)
    assert distance == 0.5


def test_distance_on_tree():
    trees = parse_newick(
        """(((A,(B,C)),((D,(E,F)))),(O1,O2));(((B,C),((D,A),(E,F))),(O1,O2));"""
    )
    """
    ORDER:
        A, B, C, D, E, F, O1, O2
        B, C, D, A, E, F, O1, O2
    """
    x = tuple(get_taxa_circular_order(trees[0]))
    y = tuple(get_taxa_circular_order(trees[1]))

    c_distance = circular_distance(x, y)
    assert c_distance == 0.375
