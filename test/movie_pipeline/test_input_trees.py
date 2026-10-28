import pytest

from brancharchitect.io import parse_newick
from brancharchitect.movie_pipeline.tree_interpolation_pipeline import (
    TreeInterpolationPipeline,
)
from brancharchitect.movie_pipeline.types import PipelineConfig
from webapp.services.msa import process_msa_data
from webapp.services.trees.frontend_builder import build_movie_data_from_result


def _ladder(names: list[str]) -> str:
    newick = names[0]
    for name in names[1:]:
        newick = f"({newick},{name})"
    return newick + ";"


def test_ladder_trees_deeper_than_the_recursion_limit_make_a_movie() -> None:
    # 1200 levels: past Python's default limit of 1000 frames.
    names = [f"t{i}" for i in range(1200)]
    moved = names[:]
    moved[3], moved[1100] = moved[1100], moved[3]
    trees = parse_newick(_ladder(names) + "\n" + _ladder(moved), force_list=True)

    # Same settings as the webapp's movie endpoint.
    result = TreeInterpolationPipeline(PipelineConfig(circular=True)).process_trees(
        trees
    )
    frames = result["interpolated_trees"]
    movie = build_movie_data_from_result(
        result=result,
        filename="ladder.nwk",
        msa_data=process_msa_data(None, num_trees=len(frames)),
    )

    assert len(frames) > 2
    assert len(movie.interpolated_trees) == len(frames)


@pytest.mark.parametrize(
    ("newick", "message"),
    [
        ("(((A,B),C),(D,E));\n(((A,B),C),(D,F));", r"missing=\['E'\] extra=\['F'\]"),
        ("((A,B),C,D);\n((A,B),C);", r"missing=\['D'\] extra=\[\]"),
        ("((A,B),(A,C));\n((A,C),(A,B));", r"repeats taxon names: \['A'\]"),
    ],
)
def test_trees_with_different_or_repeated_taxa_are_rejected(
    newick: str, message: str
) -> None:
    trees = parse_newick(newick, force_list=True)
    with pytest.raises(ValueError, match=message):
        TreeInterpolationPipeline().process_trees(trees)
