from pathlib import Path

from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.tree_interpolation.subtree_paths.planning import (
    pivot_split_registry,
)
from brancharchitect.tree_interpolation.subtree_paths.planning.pivot_split_registry import (
    PivotSplitRegistry,
    build_edge_plan,
)


def test_first_subtree_does_not_receive_unrelated_collapse_splits(monkeypatch):
    encoding = {"A": 0, "B": 1, "C": 2, "D": 3}
    tree = parse_newick("(A:1,B:1,C:1,D:1);", encoding=encoding)

    subtree_a = Partition((encoding["A"],), encoding)
    subtree_d = Partition((encoding["D"],), encoding)
    split_ab = Partition((encoding["A"], encoding["B"]), encoding)
    split_cd = Partition((encoding["C"], encoding["D"]), encoding)
    pivot = Partition(tuple(sorted(encoding.values())), encoding)

    monkeypatch.setattr(
        pivot_split_registry,
        "get_unique_splits_for_current_pivot_edge_subtree",
        lambda *_args, **_kwargs: (
            PartitionSet([split_ab, split_cd], encoding=encoding),
            PartitionSet(encoding=encoding),
        ),
    )

    plan = build_edge_plan(
        expand_splits_by_subtree={},
        collapse_splits_by_subtree={
            subtree_a: PartitionSet([split_ab], encoding=encoding),
            subtree_d: PartitionSet([split_cd], encoding=encoding),
        },
        collapse_tree=tree,
        expand_tree=tree,
        current_pivot_edge=pivot,
    )

    first_subtree = next(iter(plan))
    first_collapse_path = plan[first_subtree]["collapse"]["path_segment"]

    assert first_subtree == subtree_a
    assert split_ab in first_collapse_path
    assert split_cd not in first_collapse_path


def test_build_edge_plan_uses_visual_order_for_equal_priority_movers(monkeypatch):
    encoding = {"A": 0, "B": 1, "C": 2, "D": 3}
    tree = parse_newick("(A:1,B:1,C:1,D:1);", encoding=encoding)

    subtree_a = Partition((encoding["A"],), encoding)
    subtree_d = Partition((encoding["D"],), encoding)
    split_ab = Partition((encoding["A"], encoding["B"]), encoding)
    split_cd = Partition((encoding["C"], encoding["D"]), encoding)
    pivot = Partition(tuple(sorted(encoding.values())), encoding)

    monkeypatch.setattr(
        pivot_split_registry,
        "get_unique_splits_for_current_pivot_edge_subtree",
        lambda *_args, **_kwargs: (
            PartitionSet(encoding=encoding),
            PartitionSet([split_ab, split_cd], encoding=encoding),
        ),
    )

    plan = build_edge_plan(
        expand_splits_by_subtree={
            subtree_a: PartitionSet([split_ab], encoding=encoding),
            subtree_d: PartitionSet([split_cd], encoding=encoding),
        },
        collapse_splits_by_subtree={},
        collapse_tree=tree,
        expand_tree=tree,
        current_pivot_edge=pivot,
        subtree_order_key={subtree_d: (0,), subtree_a: (1,)},
    )

    assert next(iter(plan)) == subtree_d


def test_stepwise_planner_exposes_no_tabula_rasa_api():
    planning_dir = Path(pivot_split_registry.__file__).parent

    assert not hasattr(PivotSplitRegistry, "get_tabula_rasa_collapse_splits")
    assert not (planning_dir / "tabula_rasa_strategy.py").exists()
