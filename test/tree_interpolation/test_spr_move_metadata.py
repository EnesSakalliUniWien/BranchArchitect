import math

from brancharchitect.elements.partition import Partition
from brancharchitect.parser import parse_newick
from brancharchitect.tree_interpolation.sequential_interpolation import (
    SequentialInterpolationBuilder,
)


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
    from webapp.services.trees.frontend_builder import _serialize_tree_pair_solutions

    encoding = {"A": 0, "B": 1, "C": 2}
    pivot = Partition((0, 1, 2), encoding)
    subtree = Partition((1,), encoding)
    collapse_split = Partition((0, 1), encoding)
    expand_split = Partition((1, 2), encoding)

    serialized = _serialize_tree_pair_solutions(
        {
            "pair_0_1": {
                "jumping_subtree_solutions": {},
                "solution_to_destination_map": {},
                "solution_to_source_map": {},
                "spr_move_events": [
                    {
                        "pivot_edge": pivot,
                        "moving_subtree": subtree,
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
    assert event["moving_subtree"] == [1]
    assert event["step_range"] == [2, 5]
    assert event["collapse_path"] == [{"split": [0, 1], "branch_length": 1.25}]
    assert event["expand_path"] == [{"split": [1, 2], "branch_length": 2.5}]
    assert event["total_hops"] == 2
    assert event["total_branch_length"] == 3.75
