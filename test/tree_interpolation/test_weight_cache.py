from collections import Counter
from pathlib import Path

from brancharchitect.jumping_taxa.lattice.solvers.lattice_solver import LatticeSolver
from brancharchitect.parser import parse_newick
from brancharchitect.tree import Node
from brancharchitect.tree_interpolation.subtree_paths.execution.sequence import (
    execute_active_split_transition_sequence,
)


def test_pair_interpolation_reuses_source_and_destination_weights_per_pair(
    monkeypatch,
):
    trees = parse_newick(
        Path("test/data/current_testfiles/small_example.newick").read_text(),
        force_list=True,
    )
    source, destination = trees[:2]
    destination.initialize_split_indices(source.taxa_encoding)
    jumping_subtree_solutions, _ = LatticeSolver(
        source,
        destination,
    ).solve_iteratively()
    ordered_edges = list(jumping_subtree_solutions.keys())

    assert len(ordered_edges) > 1

    calls = Counter()
    original_to_weighted_splits = Node.to_weighted_splits

    def counted_to_weighted_splits(self):
        if self is source:
            calls["source"] += 1
        elif self is destination:
            calls["destination"] += 1
        else:
            calls["other"] += 1
        return original_to_weighted_splits(self)

    monkeypatch.setattr(Node, "to_weighted_splits", counted_to_weighted_splits)

    execute_active_split_transition_sequence(
        source_tree=source,
        destination_tree=destination,
        target_pivot_edges=ordered_edges,
        jumping_subtree_solutions=jumping_subtree_solutions,
    )

    assert calls["source"] == 1
    assert calls["destination"] == 1
