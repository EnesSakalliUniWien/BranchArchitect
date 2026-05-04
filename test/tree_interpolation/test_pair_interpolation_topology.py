from pathlib import Path

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.tree_interpolation.pair_interpolation import (
    process_tree_pair_interpolation,
)


def test_focus_pair_with_pivot_complements_reaches_destination_topology():
    lines = [
        line.strip()
        for line in Path("test/data/current_testfiles/focus.tree")
        .read_text()
        .splitlines()
        if line.strip()
    ]

    source_tree = parse_newick(lines[1])
    destination_tree = parse_newick(lines[2])

    result = process_tree_pair_interpolation(
        source_tree, destination_tree, pair_index=1
    )

    assert result.trees
    assert result.trees[-1].to_splits() == destination_tree.to_splits()
