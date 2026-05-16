import pytest

from brancharchitect.leaforder.split_analysis import (
    clear_split_pair_cache,
    get_common_splits,
)
from brancharchitect.movie_pipeline.tree_interpolation_pipeline import (
    TreeInterpolationPipeline,
)
from brancharchitect.movie_pipeline.types import PipelineConfig
from brancharchitect.parser.newick_parser import parse_newick


def test_split_pair_cache_uses_node_identity_not_tree_equality():
    clear_split_pair_cache()
    old_tree_1, old_tree_2 = parse_newick(
        "((A,B),C);((A,C),B);",
        force_list=True,
    )
    assert get_common_splits(old_tree_1, old_tree_2)

    new_tree_1, new_tree_2 = parse_newick(
        "((B,A),C);((B,C),A);",
        force_list=True,
    )
    assert get_common_splits(new_tree_1, new_tree_2)


def test_pipeline_ignores_stale_split_cache_from_previous_encoding():
    clear_split_pair_cache()
    old_tree_1, old_tree_2 = parse_newick(
        "((A,B),C);((A,C),B);",
        force_list=True,
    )
    get_common_splits(old_tree_1, old_tree_2)

    trees = parse_newick(
        "((B,A),C);((B,C),A);",
        force_list=True,
    )
    result = TreeInterpolationPipeline(
        PipelineConfig(enable_rooting=False, use_anchor_ordering=True)
    ).process_trees(trees)

    assert result["interpolated_trees"]


@pytest.fixture(autouse=True)
def _clear_split_cache():
    clear_split_pair_cache()
    yield
    clear_split_pair_cache()
