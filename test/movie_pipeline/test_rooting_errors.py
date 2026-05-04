import pytest

from brancharchitect.movie_pipeline.tree_interpolation_pipeline import (
    TreeInterpolationPipeline,
)
from brancharchitect.movie_pipeline.types import PipelineConfig
from brancharchitect.tree import Node


def test_enabled_rooting_failure_is_raised(monkeypatch):
    pipeline = TreeInterpolationPipeline(PipelineConfig(enable_rooting=True))
    tree = Node(name="A", taxa_encoding={"A": 0}, split_indices=(0,))

    def fail_rooting(trees):
        raise ValueError("rooting broke")

    monkeypatch.setattr(
        "brancharchitect.movie_pipeline.tree_interpolation_pipeline.root_trees",
        fail_rooting,
    )

    with pytest.raises(RuntimeError, match="Rooting failed"):
        pipeline._apply_rooting_if_enabled([tree])
