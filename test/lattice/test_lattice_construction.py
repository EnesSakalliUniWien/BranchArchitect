import pytest
from pathlib import Path
import json
from typing import List, Tuple
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.jumping_taxa.lattice.lattice_construction import (
    construct_sub_lattices,
    get_child_splits,
    compute_cover_elements,
    compute_unique,
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
    t2 = parse_newick(tree2_newick, t1._order)

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
    s_edges = construct_sub_lattices(t1, t2)

    if not s_edges:
        pytest.skip(f"No comparable splits found in {filename}")

    # All splits in s_edges should be in common_splits
    for edge in s_edges:
        assert edge.split in common_splits, (
            f"Split {edge.split} is not common to both trees"
        )


def test_compute_cover_elements_simple():
    """Test compute_cover_elements with a simple tree example."""
    # Create a simple tree: ((A,B),C)
    t1_newick = "((A,B),C);"
    t2_newick = "((A,C),B);"

    t1 = parse_newick(t1_newick)
    t2 = parse_newick(t2_newick, t1._order)

    # Ensure split indices are built
    if not hasattr(t1, "_split_index") or t1._split_index is None:
        t1.build_split_index()
    if not hasattr(t2, "_split_index") or t2._split_index is None:
        t2.build_split_index()

    # Get splits including leaves
    splits1 = t1.to_splits(with_leaves=True)
    splits2 = t2.to_splits(with_leaves=True)

    # Get common splits
    common_splits = splits1 & splits2

    # For root node of first tree
    root = t1
    child_splits = get_child_splits(root)

    # Create a proper common_excluding set with preserved lookup
    common_excluding = PartitionSet(common_splits, splits1.encoding, "common_excluding")

    # Test compute_cover_elements
    covers = compute_cover_elements(root, child_splits, common_excluding)

    # Debug information - print what's in each set
    print(f"Common splits: {[str(s) for s in common_splits]}")
    print(f"Cover elements: {[str(s) for s in covers]}")

    # Check each split in covers
    for ParitionSet in covers:
        for split in ParitionSet:
            assert split in common_splits, (
                f"Cover element {split} not common to both trees"
            )
            # For debugging, check specific problematic split
            if str(split) == "(A)" or str(split) == "(0,)":
                print(f"Split representation: {split}, indices: {split.indices}")
                print(
                    f"In common_splits: {any(s.indices == split.indices for s in common_splits)}"
                )
            print(type(split))
            # Convert to indices for more reliable comparison
            split_indices = split.indices
            assert any(s.indices == split_indices for s in common_splits), (
                f"Cover {split} (indices: {split_indices}) not in common splits"
            )


def test_direct_compare_tree_splits():
    """Test that compare_tree_splits only processes truly common splits."""
    # Test with a simple example from the problematic file
    t1_newick = "(((A1,A2),(X1,X2),A3),(B1,B2),(O1,O2));"
    t2_newick = "(((A1,A2,A3),(X1,X2)),(B1,B2),(O1,O2));"

    # Parse the trees
    t1 = parse_newick(t1_newick)
    t2 = parse_newick(t2_newick, t1._order)

    # Ensure split indices are properly built
    if isinstance(t1, list):
        for node in t1:
            node.build_split_index()
    else:
        t1.build_split_index()
    if isinstance(t2, list):
        for node in t2:
            node.build_split_index()
    else:
        t2.build_split_index()

    # Get splits
    splits1 = t1.to_splits(with_leaves=True)
    splits2 = t2.to_splits(with_leaves=True)

    # Get common splits
    common_splits = splits1 & splits2

    # Use compare_tree_splits
    s_edges = construct_sub_lattices(t1, t2)

    # Verify that all s_edges keys are in common_splits
    for edge in s_edges:
        assert edge.split in common_splits, f"Split {edge} is not common to both trees"

    # Verify that each HasseEdge correctly identifies common elements
    for edge in s_edges:
        # Verify lef
        # t and right cover elements are all in common_splits
        for cover_split in edge.t1_common_covers:
            for s in cover_split:
                assert s in common_splits, (
                    f"Cover split {cover_split} not common to both trees"
                )

        for cover_split in edge.t2_common_covers:
            for s in cover_split:
                assert s in common_splits, (
                    f"Cover split {cover_split} not common to both trees"
                )


def test_compute_cover_elements_preserves_common():
    """Test that compute_cover_elements preserves only common elements."""
    # Simple test case based on the problematic file
    t1_newick = "(((A1,A2),(X1,X2),A3),(B1,B2),(O1,O2));"
    t2_newick = "(((A1,A2,A3),(X1,X2)),(B1,B2),(O1,O2));"

    t1 = parse_newick(t1_newick)
    t2 = parse_newick(t2_newick, t1._order)

    # Get all splits
    splits1 = t1.to_splits(with_leaves=True)
    splits2 = t2.to_splits(with_leaves=True)

    # Get common splits
    common_splits = splits1 & splits2

    # Find a split that exists in both trees for testing
    test_split = next(iter(s for s in common_splits if len(s) > 1))
    test_node = t1.find_node_by_split(test_split)

    # Get child splits
    child_splits = get_child_splits(test_node)

    # Compute cover elements
    covers = compute_cover_elements(test_node, child_splits, common_splits)

    # All elements in the cover should be common to both trees
    for split in covers:
        for s in split:
            assert s in common_splits, f"Cover element {s} not common to both trees"


def test_atoms():
    """Test that compute_cover_elements preserves only common elements."""
    # Simple test case based on the problematic file

    encoding = {
        "A": 0,
        "B": 1,
        "C": 2,
    }

    t1_newick = "((A,B),C);"
    t2_newick = "((B,C),A);"

    t1 = parse_newick(t1_newick, encoding=encoding)
    t2 = parse_newick(t2_newick, encoding=encoding)

    D1 = get_child_splits(t1)
    D2 = get_child_splits(t2)

    t1_unique_atoms = compute_unique(t1, t2, D1, lambda ps: ps.atom())
    t2_unique_atoms = compute_unique(t2, t1, D2, lambda ps: ps.atom())

    assert t1_unique_atoms == [
        PartitionSet({(0, 1)}, encoding=encoding),
        PartitionSet({}, encoding=encoding),
    ]
    assert t2_unique_atoms == [
        PartitionSet({}, encoding=encoding),
        PartitionSet({(1, 2)}, encoding=encoding),
    ]
