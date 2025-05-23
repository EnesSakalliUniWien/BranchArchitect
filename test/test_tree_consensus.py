from brancharchitect.newick_parser import parse_newick
from brancharchitect.consensus_tree import (
    create_majority_consensus_tree,
    create_majority_consensus_tree_extended,
    create_consensus_tree,
    compatible,
    collect_splits,
)


def test_check_split_compatibility():
    assert compatible((1, 2), (3, 4))
    assert compatible((1, 2), (3, 4))
    assert compatible((1,), (3, 4))

    assert compatible((2, 4), (1, 3))
    assert compatible((1,), (2, 3, 4))

    # assert not compatible((2,), (2, 3, 4)) # TODO why should this not be compatible?
    # assert not compatible((3,), (2, 3, 4)) # TODO why should this not be compatible?
    # assert not compatible((4,), (2, 3, 4)) # TODO why should this not be compatible?
    assert not compatible((2, 3), (3, 4))
    assert not compatible((1, 3), (3, 4))


def test_count_splits():
    s = (
        "((A,(B,C)),(D,(E,F)));"
        + "((A,(B,C)),(D,(E,F)));"
        + "(((A,B),C),((D,E),F));"
        + "((A,(B,C)),(D,(E,F)));"
        + "((E,(C,D)),(A,(B,F)));"
    )

    trees = parse_newick(s)
    observed_number_of_splits = collect_splits(trees)

    observed_number_of_splits_int = {
        tuple(split.indices): freq for split, freq in observed_number_of_splits.items()
    }

    expected_number_of_splits = {
        (0, 1, 2): 0.8,
        (0, 1): 0.2,
        (1, 2): 0.6,
        (3, 4, 5): 0.8,
        (4, 5): 0.6,
        (2, 3, 4): 0.2,
        (3, 4): 0.2,
        (2, 3): 0.2,
        (1, 5): 0.2,
        (0, 1, 5): 0.2,
    }

    assert expected_number_of_splits == observed_number_of_splits_int


def test_create_consensus_tree():
    s = (
        "((A,(B,C)),(D,(E,F)));"
        + "((A,(B,C)),(D,(E,F)));"
        + "(((A,B),C),((D,E),F));"
        + "((A,(B,C)),(D,(E,F)));"
        + "((E,(C,D)),(A,(B,F)));"
    )

    trees = parse_newick(s)

    consensus_tree = create_consensus_tree(trees)
    assert "(A,B,C,D,E,F)R;" == consensus_tree.to_newick(lengths=False)


def test_create_majority_consensus_tree_extended():
    tree = create_majority_consensus_tree_extended(
        parse_newick(
            "((A,(B,C)),(D,(E,F)));"
            + "((A,(B,C)),(D,(E,F)));"
            + "(((A,B),C),((D,E),F));"
            + "((A,(B,C)),(D,(E,F)));"
            + "((E,(C,D)),(A,(B,F)));"
        )
    )

    # assert '((A,(B,C)),D,E,F)Root' == tree.to_newick(lengths=False)
    assert '(((D,(E,F))),((A,(B,C))))R;' == tree.to_newick(lengths=False)


def test_create_majority_consensus():

    tree = create_majority_consensus_tree(
        parse_newick(
            "((((A:1,B:1),C:1),D:1):1);"
            + "(((A:1,B:1,C:1),D:1):1);"
            + "(((A:1,B:1,D:1),C:1):1);"
        )
    )

    assert "(D,(A,B,C))R;" == tree.to_newick(lengths=False)
