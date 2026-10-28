from brancharchitect.io import parse_newick
from brancharchitect.movie_pipeline.tree_rooting import root_trees


def _labels_by_split(tree) -> dict[tuple[str, ...], str]:
    return {
        tuple(sorted(node.split_indices.taxa)): node.name
        for node in tree.traverse()
        if node.children and node is not tree
    }


def test_midpoint_rooting_keeps_support_on_its_split_and_full_taxon_names() -> None:
    # The long F branch puts the midpoint on it, so rooting reverses the
    # E,F edge: its 50 must stay on the EF|ABCD split, not move to F's edge.
    newick = "((A:1,B:1)90:1,(C:1,D:1)70:1,(E:1,F x:20)50:1);"
    (rooted,) = root_trees(parse_newick(newick, force_list=True))

    labels = _labels_by_split(rooted)
    assert labels[("A", "B")] == "90"
    assert labels[("C", "D")] == "70"
    assert labels[("A", "B", "C", "D")] == "50"  # same split as (E, F x)
    assert labels[("A", "B", "C", "D", "E")] == ""  # F's terminal edge
    assert sorted(leaf.name for leaf in rooted.get_leaves()) == [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F x",
    ]
