import pytest

from brancharchitect.movie_pipeline.tree_interpolation_pipeline import (
    TreeInterpolationPipeline,
)
from brancharchitect.movie_pipeline.tree_rooting import root_trees
from brancharchitect.movie_pipeline.types import PipelineConfig
from brancharchitect.parser.newick_parser import parse_newick
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


def test_rooting_reads_newick_with_explicit_format(monkeypatch):
    trees = parse_newick(
        "(A:1,B:1);\n(A:1,(B:1,C:1):1);",
        force_list=True,
        treat_zero_as_epsilon=True,
    )
    calls = []

    from brancharchitect.movie_pipeline import tree_rooting

    original_read = tree_rooting.SkbioTreeNode.read

    def read_with_format_check(file, format=None, **kwargs):
        calls.append(format)
        if format != "newick":
            raise ValueError(f"missing explicit format: {format!r}")
        return original_read(file, format=format, **kwargs)

    monkeypatch.setattr(
        tree_rooting.SkbioTreeNode,
        "read",
        staticmethod(read_with_format_check),
    )

    rooted = root_trees(trees)

    assert len(rooted) == 2
    assert calls == ["newick", "newick"]
