import importlib.util
import math
import sys
import types
from pathlib import Path

from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.parser import parse_newick
from brancharchitect.tree_interpolation.subtree_paths.execution.pivot import (
    pivot_edge_executor,
)
from brancharchitect.tree_interpolation.subtree_paths.execution.pivot.pivot_edge_executor import (
    execute_pivot_edge_interpolation,
)
from brancharchitect.tree_interpolation.sequential_interpolation import (
    SequentialInterpolationBuilder,
)
from brancharchitect.tree_interpolation.subtree_paths.planning import PivotTransitionStep


def _load_frontend_builder_serializer():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "webapp"
        / "services"
        / "trees"
        / "frontend_builder.py"
    )

    original_modules = {
        name: sys.modules.get(name)
        for name in (
            "brancharchitect.io",
            "webapp",
            "webapp.services",
            "webapp.services.serialization",
            "webapp.services.trees",
            "webapp.services.trees.movie_data",
        )
    }

    io_module = types.ModuleType("brancharchitect.io")
    io_module.serialize_tree_list_to_json = lambda trees: trees

    serialization_module = types.ModuleType("webapp.services.serialization")
    serialization_module.serialize_partition_to_indices = lambda partition: (
        list(partition.indices) if partition is not None else None
    )

    def serialize_partition_dict_to_indices(partition_dict):
        def serialize_value(value):
            if hasattr(value, "indices"):
                return list(value.indices)
            if isinstance(value, list):
                return [serialize_value(item) for item in value]
            if isinstance(value, dict):
                return serialize_partition_dict_to_indices(value)
            return value

        return {
            str(
                serialization_module.serialize_partition_to_indices(key)
            ): serialize_value(value)
            for key, value in partition_dict.items()
        }

    serialization_module.serialize_partition_dict_to_indices = (
        serialize_partition_dict_to_indices
    )

    movie_data_module = types.ModuleType("webapp.services.trees.movie_data")

    class MovieData:
        pass

    movie_data_module.MovieData = MovieData

    webapp_module = types.ModuleType("webapp")
    webapp_module.__path__ = []
    services_module = types.ModuleType("webapp.services")
    services_module.__path__ = []
    trees_module = types.ModuleType("webapp.services.trees")
    trees_module.__path__ = []

    sys.modules.update(
        {
            "brancharchitect.io": io_module,
            "webapp": webapp_module,
            "webapp.services": services_module,
            "webapp.services.serialization": serialization_module,
            "webapp.services.trees": trees_module,
            "webapp.services.trees.movie_data": movie_data_module,
        }
    )

    try:
        spec = importlib.util.spec_from_file_location(
            "_frontend_builder_under_test", module_path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module._serialize_tree_pair_solutions
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def test_tree_pair_solution_records_spr_hops_and_branch_lengths():
    encoding = {"A": 0, "B": 1, "C": 2, "D": 3}
    taxa_order = ["A", "B", "C", "D"]
    source = parse_newick(
        "((A:0.1,B:0.2):0.3,(C:0.4,D:0.5):0.6);",
        order=taxa_order,
        encoding=encoding,
    )
    destination = parse_newick(
        "((A:0.1,C:0.4):0.7,(B:0.2,D:0.5):0.8);",
        order=taxa_order,
        encoding=encoding,
    )

    sequence = SequentialInterpolationBuilder().build([source, destination])
    pair_solutions, _ = sequence.build_pair_solutions(
        sequence.get_original_tree_indices()
    )
    pair_solution = pair_solutions["pair_0_1"]

    spr_move_events = pair_solution["spr_move_events"]

    assert spr_move_events
    event = spr_move_events[0]
    assert event["total_hops"] == event["collapse_hops"] + event["expand_hops"]
    assert event["total_hops"] == len(event["collapse_path"]) + len(
        event["expand_path"]
    )
    assert event["step_range"][0] <= event["step_range"][1]

    path_segments = event["collapse_path"] + event["expand_path"]
    assert path_segments
    for segment in path_segments:
        assert isinstance(segment["split"], Partition)
        assert isinstance(segment["branch_length"], float)

    segment_total = sum(segment["branch_length"] for segment in path_segments)
    assert math.isclose(event["total_branch_length"], segment_total)
    assert math.isclose(
        event["total_branch_length"],
        event["collapse_branch_length"] + event["expand_branch_length"],
    )


def test_frontend_serializes_spr_move_events():
    _serialize_tree_pair_solutions = _load_frontend_builder_serializer()

    encoding = {"A": 0, "B": 1, "C": 2}
    pivot = Partition((0, 1, 2), encoding)
    subtree = Partition((1,), encoding)
    sibling = Partition((2,), encoding)
    collapse_split = Partition((0, 1), encoding)
    expand_split = Partition((1, 2), encoding)

    serialized = _serialize_tree_pair_solutions(
        {
            "pair_0_1": {
                "affected_subtrees_by_split": {},
                "attachment_edges_by_split": {},
                "spr_move_events": [
                    {
                        "pivot_edge": pivot,
                        "driver_subtree": subtree,
                        "highlight_group": [subtree, sibling],
                        "step_range": (2, 5),
                        "collapse_path": [
                            {
                                "split": collapse_split,
                                "branch_length": 1.25,
                            }
                        ],
                        "expand_path": [
                            {
                                "split": expand_split,
                                "branch_length": 2.5,
                            }
                        ],
                        "collapse_hops": 1,
                        "expand_hops": 1,
                        "total_hops": 2,
                        "collapse_branch_length": 1.25,
                        "expand_branch_length": 2.5,
                        "total_branch_length": 3.75,
                    }
                ],
            }
        }
    )

    event = serialized["pair_0_1"]["spr_move_events"][0]
    assert event["pivot_edge"] == [0, 1, 2]
    assert "moving_subtree" not in event
    assert "moving_subtree_group" not in event
    assert event["driver_subtree"] == [1]
    assert event["highlight_group"] == [[1], [2]]
    assert event["step_range"] == [2, 5]
    assert event["collapse_path"] == [{"split": [0, 1], "branch_length": 1.25}]
    assert event["expand_path"] == [{"split": [1, 2], "branch_length": 2.5}]
    assert event["total_hops"] == 2
    assert event["total_branch_length"] == 3.75


def test_spr_move_event_records_visual_subtree_group_from_step_tracking(monkeypatch):
    source = parse_newick("(A:1,M1:1,M2:1,B:1);")
    destination = parse_newick("(A:1,B:1,M1:1,M2:1);", encoding=source.taxa_encoding)
    destination.initialize_split_indices(source.taxa_encoding)

    encoding = source.taxa_encoding
    pivot = Partition(tuple(sorted(encoding.values())), encoding)
    mover = Partition((encoding["M1"],), encoding)
    sibling = Partition((encoding["M2"],), encoding)

    monkeypatch.setattr(
        pivot_edge_executor,
        "build_pivot_transition_plan",
        lambda *args, **kwargs: {
            mover: PivotTransitionStep(
                subtree=mover,
                collapse_path=(),
                expand_path=(),
            )
        },
    )

    def fake_build_subtree_interpolation_frames(**kwargs):
        return (
            [kwargs["interpolation_state"].deep_copy()],
            [kwargs["current_pivot_edge"]],
            kwargs["interpolation_state"],
            [[mover, sibling]],
        )

    monkeypatch.setattr(
        pivot_edge_executor,
        "build_subtree_interpolation_frames",
        fake_build_subtree_interpolation_frames,
    )

    _trees, _edges, _state, _tracker, events = execute_pivot_edge_interpolation(
        current_base_tree=source,
        destination_tree=destination,
        source_tree=source,
        current_pivot_edge=pivot,
        collapse_paths_for_pivot_edge={
            mover: PartitionSet(set(), encoding),
            sibling: PartitionSet(set(), encoding),
        },
        expand_paths_for_pivot_edge={},
        source_parent_map=None,
        dest_parent_map=None,
    )

    assert "moving_subtree" not in events[0]
    assert "moving_subtree_group" not in events[0]
    assert events[0]["driver_subtree"] == mover
    assert events[0]["highlight_group"] == [mover, sibling]


def test_execute_pivot_edge_interpolation_uses_planner_augmented_movers(monkeypatch):
    source = parse_newick("(A:1,B:1,C:1);")
    destination = parse_newick("((A:1,B:1):1,C:1);", encoding=source.taxa_encoding)

    encoding = source.taxa_encoding
    pivot = Partition(tuple(sorted(encoding.values())), encoding)
    split_ab = Partition((encoding["A"], encoding["B"]), encoding)

    original_expand_paths: dict[Partition, PartitionSet[Partition]] = {}
    seen_mover_groups: list[list[Partition]] = []

    def fake_build_pivot_transition_plan(
        expand_splits_by_subtree,
        collapse_splits_by_subtree,
        *_args,
        **_kwargs,
    ):
        expand_splits_by_subtree[pivot] = PartitionSet([split_ab], encoding=encoding)
        return {
            pivot: PivotTransitionStep(
                subtree=pivot,
                collapse_path=(),
                expand_path=(split_ab,),
            )
        }

    def fake_build_subtree_interpolation_frames(**kwargs):
        seen_mover_groups.append(kwargs["all_mover_partitions"])
        return (
            [kwargs["interpolation_state"].deep_copy()],
            [kwargs["current_pivot_edge"]],
            kwargs["interpolation_state"],
            [[pivot]],
        )

    monkeypatch.setattr(
        pivot_edge_executor,
        "build_pivot_transition_plan",
        fake_build_pivot_transition_plan,
    )
    monkeypatch.setattr(
        pivot_edge_executor,
        "build_subtree_interpolation_frames",
        fake_build_subtree_interpolation_frames,
    )

    execute_pivot_edge_interpolation(
        current_base_tree=source,
        destination_tree=destination,
        source_tree=source,
        current_pivot_edge=pivot,
        collapse_paths_for_pivot_edge={},
        expand_paths_for_pivot_edge=original_expand_paths,
        source_parent_map=None,
        dest_parent_map=None,
    )

    assert seen_mover_groups == [[pivot]]
    assert original_expand_paths == {}
