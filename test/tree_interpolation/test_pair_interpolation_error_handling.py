"""Tests for domain-specific error handling when a tree pair's lattice solution
cannot be computed, either via a precomputed value or the synchronous retry."""
from pathlib import Path

import pytest

from brancharchitect.jumping_taxa.exceptions import PairLatticeSolveError
from brancharchitect.jumping_taxa.lattice.solvers.lattice_solver import LatticeSolver
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.tree_interpolation.pair_interpolation import (
    process_tree_pair_interpolation,
)


def _load_trees():
    lines = [
        line.strip()
        for line in Path("test/data/current_testfiles/focus.tree").read_text().splitlines()
        if line.strip()
    ]
    return parse_newick(lines[1]), parse_newick(lines[2])


def test_unsolvable_pair_raises_domain_error_naming_the_pair(monkeypatch):
    """When no precomputed solution is available and the synchronous lattice retry
    fails, the caller gets a domain-specific error naming the pair -- not whatever
    raw exception LatticeSolver happened to raise internally."""
    source_tree, destination_tree = _load_trees()

    def _boom(self):
        raise RuntimeError("simulated internal lattice failure")

    monkeypatch.setattr(LatticeSolver, "solve_iteratively", _boom)

    with pytest.raises(PairLatticeSolveError, match="pair 4-5"):
        process_tree_pair_interpolation(
            source_tree, destination_tree, precomputed_solutions=None, pair_index=4
        )
