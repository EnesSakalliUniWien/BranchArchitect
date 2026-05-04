from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.leaforder import rotation_functions
from brancharchitect.parser.newick_parser import parse_newick


def test_optimize_unique_splits_default_rotated_splits_is_per_call(monkeypatch):
    tree1 = parse_newick("((A:1,B:1):1,C:1);")
    tree2 = parse_newick("((A:1,B:1):1,C:1);", encoding=tree1.taxa_encoding)
    split = Partition(
        (tree1.taxa_encoding["A"], tree1.taxa_encoding["B"]), tree1.taxa_encoding
    )
    initial_lengths = []

    def fake_get_unique_splits(tree_one, tree_two):
        return PartitionSet({split}, encoding=tree_one.taxa_encoding)

    def fake_optimize_splits(
        tree, splits_to_optimize, destination_order, rotated_splits
    ):
        initial_lengths.append(len(rotated_splits))
        rotated_splits.add(split)
        return True

    monkeypatch.setattr(rotation_functions, "get_unique_splits", fake_get_unique_splits)
    monkeypatch.setattr(rotation_functions, "optimize_splits", fake_optimize_splits)

    rotation_functions.optimize_unique_splits(tree1, tree2, tree1.get_current_order())
    rotation_functions.optimize_unique_splits(tree1, tree2, tree1.get_current_order())

    assert initial_lengths == [0, 0]
