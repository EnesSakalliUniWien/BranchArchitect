from brancharchitect.newick_parser import parse_newick
from brancharchitect.distances import (
    robinson_foulds_distance,
    weighted_robinson_foulds_distance,
    calculate_along_trajectory,
)
import pytest


def test_splits_are_sorted():
    s = "(A:1,(B:1,C:1):1);"
    t1 = parse_newick(s)
    for split_indices in t1.to_splits():
        assert tuple(sorted(split_indices)) == split_indices


def test_collect_splits():
    s = "(A:1,(B:1,C:1):1);"
    t1 = parse_newick(s)
    expected_splits = sorted([tuple([0, 1, 2]), tuple([1, 2])])

    # Collect splits and sort the observed lists
    observed_splits = sorted(list(t1.to_splits()))

    assert expected_splits == observed_splits


trees_and_distance = [
    ("(A:1,(B:1,C:1):1);(B:1,(A:1,C:1):1);", 1, 1 / 2, 2.0),
    ("((A:1,B:1):1,(C:1,D:1):1);((C:1,A:1):1,(B:1,D:1):1);", 2, 2 / 4, 4.0),
    ("((A:1,B:1):1,(C:1,D:1):1);((A:1,B:1):1,(C:1,D:1):1);", 0, 0.0, 0),
]


@pytest.fixture(params=trees_and_distance)
def tree_and_rfd(request):
    s, rfd, rrfd, wrfd = request.param
    t1, t2 = parse_newick(s)
    return t1, t2, rfd


@pytest.fixture(params=trees_and_distance)
def tree_and_rrfd(request):
    s, rfd, rrfd, wrfd = request.param
    t1, t2 = parse_newick(s)
    return t1, t2, rrfd


@pytest.fixture(params=trees_and_distance)
def tree_and_wrfd(request):
    s, rfd, rrfd, wrfd = request.param
    t1, t2 = parse_newick(s)
    return t1, t2, wrfd


trajectories_and_distances = [
    (
        "((A:1,B:1):1,(C:1,D:1):1);((A:1,B:1):1,(C:1,D:1):1);((A:1,B:1):1,(C:1,D:1):1);",
        [0, 0],
    ),
    (
        "((A:1,B:1):1,(C:1,D:1):1);((C:1,A:1):1,(B:1,D:1):1);((C:1,A:1):1,(B:1,D:1):1);",
        [4.0, 0],
    ),
    (
        "((A:1,B:1):1,(C:1,D:1):1);((C:1,A:1):1,(B:1,D:1):1);((A:1,B:1):1,(C:1,D:1):1);",
        [4.0, 4.0],
    ),
]


@pytest.fixture(params=trajectories_and_distances)
def trajectory_and_distance(request):
    s, dists = request.param
    trees = parse_newick(s)
    return trees, dists


def test_distances(tree_and_rfd):
    t1, t2, expected_distance = tree_and_rfd
    relative_distance = robinson_foulds_distance(t1, t2)
    assert relative_distance == expected_distance


def test_distances_w(tree_and_wrfd):
    t1, t2, expected_distance = tree_and_wrfd
    relative_distance = weighted_robinson_foulds_distance(t1, t2)
    assert relative_distance == expected_distance


def test_trajectory_distances(trajectory_and_distance):
    trees, expected_distances = trajectory_and_distance
    observed_distances = calculate_along_trajectory(
        trees, weighted_robinson_foulds_distance
    )
    assert observed_distances == expected_distances


def test_pair_weights_distances_test_one():
    s = "((A:1,B:1):1,(C:1,D:1):1);" + "((A:1,B:1):1,(C:1,D:1):1);"
    trees = parse_newick(s)
    observed_distance = weighted_robinson_foulds_distance(trees[0], trees[1])
    assert observed_distance == 0


def test_trajectory_distances_test_two():
    s = (
        "((A:1,B:1):1,(C:1,D:1):1);"
        + "((C:1,A:1):1,(B:1,D:1):1);"
        + "((C:1,A:1):1,(B:1,D:1):1);"
    )
    trees = parse_newick(s)
    expected_distances = [4, 0]
    observed_relative_distances = calculate_along_trajectory(
        trees, weighted_robinson_foulds_distance
    )

    assert expected_distances == observed_relative_distances
