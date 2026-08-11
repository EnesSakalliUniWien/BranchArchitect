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


def test_weighted_splits_refresh_after_branch_length_mutation():
    """A cached weight must not survive a mutation of the branch it describes."""
    tree = parse_newick("((A:1.0,B:2.0)ab:3.0,C:4.0);", force_list=True)[0]
    target = next(node for node in tree.traverse() if node.name == "B")

    assert tree.to_weighted_splits()[target.split_indices] == 2.0

    target.length = 9.0

    assert tree.to_weighted_splits()[target.split_indices] == 9.0


def test_weighted_splits_refresh_for_every_ancestor_of_the_mutated_branch():
    """Every ancestor caches the mutated branch, so every ancestor must be cleared."""
    tree = parse_newick("(((A:1.0,B:2.0)ab:3.0,C:4.0)abc:5.0,D:6.0);", force_list=True)[
        0
    ]
    target = next(node for node in tree.traverse() if node.name == "B")
    ancestors = [node for node in tree.traverse() if node.name in {"ab", "abc"}]

    assert len(ancestors) == 2
    for ancestor in [tree, *ancestors]:
        assert ancestor.to_weighted_splits()[target.split_indices] == 2.0

    target.length = 9.0

    for ancestor in [tree, *ancestors]:
        assert ancestor.to_weighted_splits()[target.split_indices] == 9.0


def test_weighted_splits_cache_is_reused_when_no_length_changes():
    """The cache must still be reused; invalidation is mutation-driven only."""
    tree = parse_newick("((A:1.0,B:2.0)ab:3.0,C:4.0);", force_list=True)[0]
    target = next(node for node in tree.traverse() if node.name == "B")

    first = tree.to_weighted_splits()
    target.length = target.length

    assert tree.to_weighted_splits() is first
