# Enhanced tests for component_distance.py
from typing import Any, List
from numpy import dtype
from numpy._typing._array_like import NDArray
import pytest
from brancharchitect.elements.partition import Partition
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.distances.component_distance import (
    component_distance,
    jump_path_distance,
    extract_leaf_order,
    get_lattice_solution_sizes,
)

# Additional imports for enhanced coverage
from brancharchitect.distances.component_distance import (
    clear_jump_path_cache,
    jump_distance,
    jump_path,
    calculate_component_distance_matrix,
)
from brancharchitect.tree import Node


# --- Edge Case Tests ---
def test_empty_trees():
    with pytest.raises(ValueError):
        extract_leaf_order([])


def test_empty_component_list():
    trees: Node | List[Node] = parse_newick("(((A,B),(C,D)),E);")
    if not isinstance(trees, list):
        trees = [trees]
    result: List[float] = component_distance(trees[0], trees[0], [])
    assert result == []


def test_trivial_components():
    trees: Node | List[Node] = parse_newick("(((A,B),(C,D)),E);")
    if not isinstance(trees, list):
        trees = [trees]
    # All leaves and single leaf components
    all_leaves = tuple(sorted([leaf.name for leaf in trees[0].get_leaves()]))
    single_leaf: tuple[str | None] = (trees[0].get_leaves()[0].name,)
    # Should be ignored or return 0
    result = component_distance(trees[0], trees[0], [all_leaves, single_leaf])
    assert result == [0, 0] or result == []


def test_component_not_in_tree():
    trees: Node | List[Node] = parse_newick("(((A,B),(C,D)),E);")
    if not isinstance(trees, list):
        trees = [trees]
    # 'Z' is not in the tree
    with pytest.raises(ValueError):
        component_distance(trees[0], trees[0], [("Z",)])
    # Should raise ValueError because 'Z' is not in the tree


# --- Error Handling ---
def test_invalid_component_type():
    trees = parse_newick("(((A,B),(C,D)),E);")
    if not isinstance(trees, list):
        trees = [trees]
    with pytest.raises(Exception):
        component_distance(trees[0], trees[0], [123])


def test_no_overlap_leaves():
    t1 = parse_newick("(((A,B),(C,D)),E);")
    t2 = parse_newick("(((X,Y),(Z,W)),Q);")
    if not isinstance(t1, list):
        t1 = [t1]
    if not isinstance(t2, list):
        t2 = [t2]
    # No overlap in leaves
    result = component_distance(t1[0], t2[0], [("A",)])
    assert isinstance(result, list)


# --- Polytomy and Missing Data ---
def test_polytomy():
    trees = parse_newick("(A,B,C,D);")
    if not isinstance(trees, list):
        trees = [trees]
    result = component_distance(trees[0], trees[0], [("A",)])
    assert result == [0]


def test_missing_branch_lengths_weighted():
    trees = parse_newick("(((A,B),(C,D)),E);")
    if not isinstance(trees, list):
        trees = [trees]
    result = component_distance(trees[0], trees[0], [("A",)], weighted=True)
    assert result == [0.0]


# --- Utility Function Tests ---
def test_extract_leaf_order():
    trees = parse_newick("(((A,B),(C,D)),E);")
    if not isinstance(trees, list):
        trees = [trees]
    order = extract_leaf_order(trees)
    assert set(order) == set(["A", "B", "C", "D", "E"])


def test_get_lattice_solution_sizes():
    trees = parse_newick("(((A,B),(C,D)),E);(((A,B),(C,D)),E);")
    if not isinstance(trees, list):
        trees = [trees]
    sizes = get_lattice_solution_sizes(trees[0], trees[1], extract_leaf_order(trees))
    assert isinstance(sizes, list)


# --- Randomized Test ---
def test_randomized_components():
    trees = parse_newick("(((A,B),(C,D)),E);(((A,B),(C,D)),E);")
    if not isinstance(trees, list):
        trees = [trees]
    import random

    leaves = [leaf.name for leaf in trees[0].get_leaves()]
    comps = [tuple(random.sample(leaves, k)) for k in range(2, len(leaves))]
    result = component_distance(trees[0], trees[1], comps)
    assert isinstance(result, list)
    assert len(result) == len(comps)


# =============================
# Enhanced Coverage for All Functions
# =============================
def test_clear_jump_path_cache():
    # Fill cache by calling jump_path
    trees = parse_newick("(((A,B),(C,D)),E);(((A,B),(C,D)),E);")
    if not isinstance(trees, list):
        trees = [trees]
    component = ("A", "B")
    # Use PartitionSet from tree2
    reference = trees[1].to_splits()
    # Call jump_path to fill cache
    path1 = jump_path(trees[0], reference, trees[0].names_to_partition(component))
    assert path1 is not None
    # Clear cache
    clear_jump_path_cache()
    # After clearing, cache should be empty (indirectly tested by no error on next call)
    path2 = jump_path(trees[0], reference, trees[0].names_to_partition(component))
    assert path2 is not None


def test_jump_distance_adapter():
    trees = parse_newick("(((A,B),(C,D)),E);(((A,B),(C,D)),E);")
    if not isinstance(trees, list):
        trees = [trees]
    component = ("A", "B")
    reference = trees[1].to_splits()
    # Unweighted
    dist = jump_distance(trees[0], reference, component)
    assert isinstance(dist, float)
    # Weighted
    dist_w = jump_distance(trees[0], reference, component, weighted=True)
    assert isinstance(dist_w, float)


def test_jump_path_public():
    trees = parse_newick("(((A,B),(C,D)),E);(((A,B),(C,D)),E);")
    if not isinstance(trees, list):
        trees = [trees]
    component = ("A", "B")
    reference = trees[1].to_splits()
    partition = trees[0].names_to_partition(component)
    path = jump_path(trees[0], reference, partition)
    assert isinstance(path, list)


def test_jump_path_distance_adapter():
    trees = parse_newick("(((A,B),(C,D)),E);(((A,B),(C,D)),E);")
    if not isinstance(trees, list):
        trees = [trees]
    components = [("A", "B"), ("C", "D")]
    # Unweighted
    dists = jump_path_distance(trees[0], trees[1], components)
    assert isinstance(dists, list)
    # Weighted
    dists_w = jump_path_distance(trees[0], trees[1], components, weighted=True)
    assert isinstance(dists_w, list)


def test_calculate_component_distance_matrix():
    trees: Node | List[Node] = parse_newick(
        "(((A,B),(C,D)),E);(((A,B),(C,D)),E);(((A,B),(C,D)),E);"
    )
    if not isinstance(trees, list):
        trees = [trees]
    # Use the same components for all pairs
    comps: List[List[Partition]] = [
        [
            trees[0].names_to_partition(("A", "B")),
            trees[0].names_to_partition(("C", "D")),
        ]
    ] * len(trees)
    matrix = calculate_component_distance_matrix(trees, comps, weighted=False)
    assert matrix.shape[0] == len(trees)
    assert matrix.shape[1] == len(trees)
    # Weighted
    matrix_w: ndarray[Any, dtype[Any]] = calculate_component_distance_matrix(
        trees, comps, weighted=True
    )
    assert matrix_w.shape == matrix.shape
