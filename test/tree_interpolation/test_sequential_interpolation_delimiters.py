from brancharchitect.parser import parse_newick
from brancharchitect.tree_interpolation.sequential_interpolation import (
    SequentialInterpolationBuilder,
)


def test_final_zero_interpolation_pair_uses_destination_delimiter():
    encoding = {"A": 0, "B": 1, "C": 2, "D": 3}
    order = ["A", "B", "C", "D"]
    source = parse_newick(
        "((A:1,B:2):3,(C:4,D:5):6):7;",
        order=order,
        encoding=encoding,
    )
    destination = parse_newick(
        "((A:10,B:20):30,(C:40,D:50):60):70;",
        order=order,
        encoding=encoding,
    )

    result = SequentialInterpolationBuilder().build([source, destination])

    assert len(result.interpolated_trees) == 2
    assert result.pair_interpolated_tree_counts == [0]
    assert result.interpolated_trees[0].to_newick() == source.to_newick()
    assert result.interpolated_trees[-1].to_newick() == destination.to_newick()
