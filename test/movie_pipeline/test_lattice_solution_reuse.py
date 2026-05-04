from brancharchitect.jumping_taxa.lattice.solvers.lattice_solver import LatticeSolver
from brancharchitect.movie_pipeline import tree_interpolation_pipeline
from brancharchitect.movie_pipeline.tree_interpolation_pipeline import (
    TreeInterpolationPipeline,
)
from brancharchitect.movie_pipeline.types import PipelineConfig
from brancharchitect.parser.newick_parser import parse_newick


def test_pipeline_solves_lattice_once_per_tree_pair(monkeypatch):
    trees = parse_newick(
        "".join(
            [
                "((A:1,B:1):1,C:1);",
                "(A:1,(B:1,C:1):1);",
                "((A:1,C:1):1,B:1);",
            ]
        )
    )
    for tree in trees[1:]:
        tree.initialize_split_indices(trees[0].taxa_encoding)

    monkeypatch.setattr(tree_interpolation_pipeline.sys, "frozen", True, raising=False)

    original_solve = LatticeSolver.solve_iteratively
    solve_calls = 0

    def counted_solve(self, *args, **kwargs):
        nonlocal solve_calls
        solve_calls += 1
        return original_solve(self, *args, **kwargs)

    monkeypatch.setattr(LatticeSolver, "solve_iteratively", counted_solve)

    pipeline = TreeInterpolationPipeline(
        PipelineConfig(enable_rooting=False, use_anchor_ordering=True)
    )
    pipeline.process_trees(trees)

    assert solve_calls == len(trees) - 1


def test_single_pair_precompute_does_not_start_joblib(monkeypatch):
    trees = parse_newick("((A:1,B:1):1,C:1);" "(A:1,(B:1,C:1):1);")
    trees[1].initialize_split_indices(trees[0].taxa_encoding)

    monkeypatch.setattr(tree_interpolation_pipeline.sys, "frozen", False, raising=False)

    def fail_parallel(*args, **kwargs):
        raise AssertionError("joblib should not be used for one tree pair")

    def fake_solve_pair(tree_one, tree_two):
        return {}, None

    monkeypatch.setattr(tree_interpolation_pipeline, "Parallel", fail_parallel)
    monkeypatch.setattr(
        tree_interpolation_pipeline, "_parallel_solve_pair", fake_solve_pair
    )

    pipeline = TreeInterpolationPipeline(PipelineConfig(enable_rooting=False))

    assert pipeline._precompute_pair_solutions(trees) == [{}]
