# Enhanced tests for component_distance.py
import pytest
from brancharchitect.newick_parser import parse_newick
from brancharchitect.component_distance import (
    component_distance, jump_path_distance, extract_leaf_order, extract_components,
    get_lattice_solution_sizes, validate_component, compute_combined_distances
)
import pandas as pd

# --- Edge Case Tests ---
def test_empty_trees():
    with pytest.raises(ValueError):
        extract_leaf_order([])

def test_empty_component_list():
    trees = parse_newick("(((A,B),(C,D)),E);")
    if not isinstance(trees, list):
        trees = [trees]
    result = component_distance(trees[0], trees[0], [])
    assert result == []

def test_trivial_components():
    trees = parse_newick("(((A,B),(C,D)),E);")
    if not isinstance(trees, list):
        trees = [trees]
    # All leaves and single leaf components
    all_leaves = tuple(sorted([leaf.name for leaf in trees[0].get_leaves()]))
    single_leaf = (trees[0].get_leaves()[0].name,)
    # Should be ignored or return 0
    result = component_distance(trees[0], trees[0], [all_leaves, single_leaf])
    assert result == [0, 0] or result == []

def test_component_not_in_tree():
    trees = parse_newick("(((A,B),(C,D)),E);")
    if not isinstance(trees, list):
        trees = [trees]
    # 'Z' is not in the tree
    with pytest.raises(ValueError):
        component_distance(trees[0], trees[0], [('Z',)])
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
    result = component_distance(t1[0], t2[0], [('A',)])
    assert isinstance(result, list)

# --- Polytomy and Missing Data ---
def test_polytomy():
    trees = parse_newick("(A,B,C,D);")
    if not isinstance(trees, list):
        trees = [trees]
    result = component_distance(trees[0], trees[0], [('A',)])
    assert result == [0]

def test_missing_branch_lengths_weighted():
    trees = parse_newick("(((A,B),(C,D)),E);")
    if not isinstance(trees, list):
        trees = [trees]
    result = component_distance(trees[0], trees[0], [('A',)], weighted=True)
    assert result == [0.0]

# --- Utility Function Tests ---
def test_extract_leaf_order():
    trees = parse_newick("(((A,B),(C,D)),E);")
    if not isinstance(trees, list):
        trees = [trees]
    order = extract_leaf_order(trees)
    assert set(order) == set(['A','B','C','D','E'])

def test_extract_components():
    trees = parse_newick("(((A,B),(C,D)),E);")
    if not isinstance(trees, list):
        trees = [trees]
    comps = extract_components(trees)
    # Should not include trivial splits
    for c in comps:
        assert 1 < len(c) < len(trees[0].get_leaves())

def test_get_lattice_solution_sizes():
    trees = parse_newick("(((A,B),(C,D)),E);(((A,B),(C,D)),E);")
    if not isinstance(trees, list):
        trees = [trees]
    sizes = get_lattice_solution_sizes(trees[0], trees[1], extract_leaf_order(trees))
    assert isinstance(sizes, list)

# --- validate_component ---
def test_validate_component():
    trees = parse_newick("(((A,B),(C,D)),E);(((A,B),(C,D)),E);")
    if not isinstance(trees, list):
        trees = [trees]
    assert validate_component(('A','B'), trees[0], trees[1])
    assert not validate_component(('A','Z'), trees[0], trees[1])

# --- compute_combined_distances ---
def test_compute_combined_distances():
    trees = parse_newick("(((A,B),(C,D)),E);(((A,B),(C,D)),E);")
    if not isinstance(trees, list):
        trees = [trees]
    comps = extract_components(trees)
    leaf_order = extract_leaf_order(trees)
    df = compute_combined_distances(trees, comps, leaf_order, lambda t1, t2, c: 0)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty

# --- Stress Test ---
def test_large_tree_stress():
    # 10 leaves, fully balanced
    newick = "((((A,B),(C,D)),((E,F),(G,H))),((I,J),(K,L)));"
    trees = parse_newick(newick + newick)
    if not isinstance(trees, list):
        trees = [trees]
    comps = extract_components(trees)
    result = component_distance(trees[0], trees[1], comps)
    assert isinstance(result, list)
    assert len(result) == len(comps)

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