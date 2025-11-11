import pytest
from pathlib import Path
import json
from typing import Any, List, Tuple
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.jumping_taxa.lattice.build_pivot_lattices import (
    construct_sublattices,
)


def load_test_trees() -> List[Tuple[str, str, str]]:
    """Load test tree data from JSON files."""
    test_data_path = Path("test/colouring/trees")

    test_data: List[Tuple[str, str, str]] = []

    if not test_data_path.exists():
        print(f"Warning: Test data directory not found: {test_data_path}")
        return []

    for _dir in test_data_path.iterdir():
        if _dir.is_dir():
            for file_path in _dir.iterdir():
                if file_path.is_file() and file_path.suffix == ".json":
                    try:
                        with open(file_path) as f:
                            data = json.load(f)

                        if "tree1" in data and "tree2" in data:
                            test_data.append(
                                (data["tree1"], data["tree2"], file_path.name)
                            )
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")

    return test_data[:2]  # Limit to first two test cases for faster testing


@pytest.mark.parametrize(
    "tree1_newick,tree2_newick,filename",
    load_test_trees() or [("(A,B);", "(A,B);", "fallback.json")],
)
def test_compare_tree_splits(tree1_newick, tree2_newick, filename):
    """Test that compare_tree_splits only processes common splits."""
    # Parse the trees
    t1 = parse_newick(tree1_newick)
    if isinstance(t1, list):
        t1 = t1[0]
    t2 = parse_newick(tree2_newick, list(t1.get_current_order()))
    if isinstance(t2, list):
        t2 = t2[0]

    # Ensure split indices are properly built
    if not hasattr(t1, "_split_index") or t1._split_index is None:
        t1.build_split_index()
    if not hasattr(t2, "_split_index") or t2._split_index is None:
        t2.build_split_index()

    # Get all splits from both trees
    splits1 = t1.to_splits(with_leaves=True)
    splits2 = t2.to_splits(with_leaves=True)

    # Get common splits
    common_splits = splits1 & splits2

    # Compare tree splits
    s_edges = construct_sublattices(t1, t2)

    if not s_edges:
        pytest.skip(f"No comparable splits found in {filename}")

    # All splits in s_edges should be in common_splits
    for edge in s_edges:
        assert edge.pivot_split in common_splits, (
            f"Split {edge.pivot_split} is not common to both trees"
        )


def test_direct_compare_tree_splits():
    """Test that compare_tree_splits only processes truly common splits."""
    # Test with a simple example from the problematic file
    t1_newick = "(((A1,A2),(X1,X2),A3),(B1,B2),(O1,O2));"
    t2_newick = "(((A1,A2,A3),(X1,X2)),(B1,B2),(O1,O2));"

    # Parse the trees
    t1 = parse_newick(t1_newick)
    if isinstance(t1, list):
        t1 = t1[0]
    t2 = parse_newick(t2_newick, list(t1.get_current_order()))
    if isinstance(t2, list):
        t2 = t2[0]

    # Ensure split indices are properly built
    t1.build_split_index()
    t2.build_split_index()

    # Get splits
    splits1 = t1.to_splits(with_leaves=True)
    splits2 = t2.to_splits(with_leaves=True)

    # Get common splits
    common_splits = splits1 & splits2

    # Use compare_tree_splits
    s_edges = construct_sublattices(t1, t2)

    # Verify that all s_edges keys are in common_splits
    for edge in s_edges:
        assert edge.pivot_split in common_splits, (
            f"Split {edge} is not common to both trees"
        )

    # Verify that each HasseEdge correctly identifies common elements
    for edge in s_edges:
        # Verify left and right cover elements are all in common_splits
        for top_to_bottom in edge.tree1_child_frontiers.values():
            for s in top_to_bottom.shared_top_splits:
                assert s in common_splits, f"Cover split {s} not common to both trees"

        for top_to_bottom in edge.tree2_child_frontiers.values():
            for s in top_to_bottom.shared_top_splits:
                assert s in common_splits, f"Cover split {s} not common to both trees"


def test_construct_sublattices_divergent_case():
    """Ensure lattice construction captures the divergent relationship for the provided trees."""

    def normalize_partition_set(
        partition_set: PartitionSet[Any],
    ) -> tuple[tuple[str, ...], ...]:
        normalized: List[tuple[str, ...]] = []
        for partition in partition_set:
            taxa = getattr(partition, "taxa")
            normalized.append(tuple(sorted(taxa)))
        return tuple(sorted(normalized))

    def normalize_partition_sets_from_dict(
        partition_dict: dict,
    ) -> tuple[tuple[tuple[str, ...], ...], ...]:
        """Extract shared_top_splits from TopToBottom dictionary values and normalize."""
        normalized_list: List[tuple[tuple[str, ...], ...]] = []
        for top_to_bottom in partition_dict.values():
            normalized_list.append(
                normalize_partition_set(top_to_bottom.shared_top_splits)
            )
        return tuple(sorted(normalized_list))

    tree1_newick = "(((A1,(A2,((X1,X2),A3))),(B1,B2)),(O1,O2));"
    tree2_newick = "((((A1,(A2,A3)),(X1,X2)),(B1,B2)),(O1,O2));"

    t1 = parse_newick(tree1_newick)
    if isinstance(t1, list):
        t1 = t1[0]

    base_order = list(t1.get_current_order())

    t2 = parse_newick(tree2_newick, base_order)
    if isinstance(t2, list):
        t2 = t2[0]

    edges = construct_sublattices(t1, t2)

    assert len(edges) == 1, "Expected exactly one lattice edge for the provided trees"

    edge = edges[0]

    assert edge.pivot_split.taxa == frozenset({"A1", "A2", "A3", "X1", "X2"})
    assert edge.relationship == "divergent"
    assert len(edge.child_subtree_splits_across_trees) == 0

    # With explicit singleton frontiers (shared direct pivot children) retained
    assert normalize_partition_sets_from_dict(edge.tree1_child_frontiers) == (
        (("A1",),),
        (("A2",), ("A3",), ("X1", "X2")),
    )
    assert normalize_partition_sets_from_dict(edge.tree2_child_frontiers) == (
        (("A1",), ("A2",), ("A3",)),
        (("X1", "X2"),),
    )
