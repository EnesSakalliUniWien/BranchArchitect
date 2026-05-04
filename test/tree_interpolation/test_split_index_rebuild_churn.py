from brancharchitect.elements.partition import Partition
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.tree_interpolation.topology_ops.collapse import (
    create_collapsed_consensus_tree,
)
from brancharchitect.tree_interpolation.topology_ops.expand import (
    _find_split_application_parent,
    apply_split_simple,
    create_subtree_grafted_tree,
    execute_expand_path,
)
from brancharchitect.tree_interpolation.topology_ops.weights import (
    apply_zero_branch_lengths,
)
from brancharchitect.elements.partition_set import PartitionSet


def test_apply_split_leaves_split_index_available():
    tree = parse_newick("(A:1,B:1,C:1,D:1);")
    split = Partition(
        (tree.taxa_encoding["A"], tree.taxa_encoding["B"]), tree.taxa_encoding
    )

    apply_split_simple(split, tree)

    assert tree._split_index is not None
    assert tree.find_node_by_split(split) is not None


def test_graft_leaves_split_index_available():
    tree = parse_newick("(A:1,B:1,C:1,D:1);")
    split = Partition(
        (tree.taxa_encoding["A"], tree.taxa_encoding["B"]), tree.taxa_encoding
    )

    grafted = create_subtree_grafted_tree(tree, [split], copy=True)

    assert grafted._split_index is not None
    assert grafted.find_node_by_split(split) is not None


def test_collapse_leaves_split_index_available():
    source = parse_newick("((A:1,B:1):1,(C:1,D:1):1);")
    destination = parse_newick("(A:1,B:1,C:1,D:1);", encoding=source.taxa_encoding)
    split = Partition(
        (source.taxa_encoding["A"], source.taxa_encoding["B"]), source.taxa_encoding
    )
    zeroed = apply_zero_branch_lengths(
        source, PartitionSet([split], source.taxa_encoding)
    )

    collapsed = create_collapsed_consensus_tree(
        zeroed,
        zeroed.split_indices,
        destination_tree=destination,
        copy=True,
    )

    assert collapsed._split_index is not None
    assert collapsed.find_node_by_split(collapsed.split_indices) is collapsed


def test_expand_parent_lookup_returns_narrowest_parent():
    tree = parse_newick("(((A:1,B:1):1,(C:1,D:1):1,E:1):1,F:1);")
    encoding = tree.taxa_encoding
    split_abcd = Partition(
        (
            encoding["A"],
            encoding["B"],
            encoding["C"],
            encoding["D"],
        ),
        encoding,
    )
    split_abcde = Partition(
        (
            encoding["A"],
            encoding["B"],
            encoding["C"],
            encoding["D"],
            encoding["E"],
        ),
        encoding,
    )

    tree.build_split_index()
    assert tree._split_index is not None

    parent = _find_split_application_parent(split_abcd, tree._split_index)

    assert parent is not None
    assert parent.split_indices == split_abcde


def test_expand_path_updates_lookup_for_nested_splits():
    tree = parse_newick("(A:1,B:1,C:1,D:1,E:1);")
    encoding = tree.taxa_encoding
    split_abcd = Partition(
        (
            encoding["A"],
            encoding["B"],
            encoding["C"],
            encoding["D"],
        ),
        encoding,
    )
    split_abc = Partition(
        (
            encoding["A"],
            encoding["B"],
            encoding["C"],
        ),
        encoding,
    )

    execute_expand_path(tree, [split_abc, split_abcd])

    assert tree.find_node_by_split(split_abcd) is not None
    assert tree.find_node_by_split(split_abc) is not None
