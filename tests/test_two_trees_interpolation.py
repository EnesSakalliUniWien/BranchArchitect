from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.tree_interpolation.active_changing_split_interpolation import (
    build_active_changing_split_interpolation_sequence,
)


def test_two_tree_interpolation_runs_and_finishes():
    tree1 = "(((((A,B),(C,D)),((E,F),G)),((I,J),L)),O);"
    tree2 = "((((A,B),D),((((E,F),G),((I,J),L)),C)),O);"

    t1 = parse_newick(tree1, force_list=True, treat_zero_as_epsilon=True)[0]
    t2 = parse_newick(tree2, force_list=True, treat_zero_as_epsilon=True)[0]

    result = build_active_changing_split_interpolation_sequence(t1, t2, tree_index=0)

    # Basic sanity checks: should produce a non-empty sequence
    assert result is not None
    assert isinstance(result.trees, list)
    assert len(result.trees) >= 1

    # Names and tracking should align in length with trees
    assert len(result.names) == len(result.trees)
    assert len(result.s_edge_tracking) == len(result.trees)

