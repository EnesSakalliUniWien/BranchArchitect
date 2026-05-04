from brancharchitect.elements.partition import Partition
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.tree_interpolation.topology_ops.collapse import (
    create_collapsed_consensus_tree,
)
from brancharchitect.tree_interpolation.topology_ops.expand import (
    apply_split_simple,
    create_subtree_grafted_tree,
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
