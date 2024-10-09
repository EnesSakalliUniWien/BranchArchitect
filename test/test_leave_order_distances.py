from brancharchitect.leaf_order_distances import create_ranks, circular_distance
from brancharchitect.newick_parser import parse_newick
from brancharchitect.consensus_tree import get_taxa_circular_order


def test_assign_ranks():
    x = ["b", "c", "d", "a"]
    y = ["b", "a", "d", "c"]
    rank_x, rank_y = create_ranks(x, y)
    assert rank_x == [0, 1, 2, 3]
    assert rank_y == [0, 3, 2, 1]


def test_circular_distance():
    x = ["b", "c", "d", "a"]
    y = ["b", "a", "d", "c"]
    distance = circular_distance(x, y)
    assert distance == 4.0


def test_distance_on_tree():
    trees = parse_newick(
        """(((A,(B,C)),((D,(E,F)))),(O1,O2));(((B,C),((D,A),(E,F))),(O1,O2));"""
    )
    """
    ORDER:
        A, B, C, D, E, F, O1, O2
        B, C, D, A, E, F, O1, O2    
    """
    x = get_taxa_circular_order(trees[0])
    y = get_taxa_circular_order(trees[1])

    c_distance = circular_distance(x, y)
    assert c_distance == 12
