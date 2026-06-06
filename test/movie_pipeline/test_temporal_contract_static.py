from pathlib import Path

from brancharchitect.elements.partition import Partition
from brancharchitect.movie_pipeline.temporal_contract import build_temporal_contract
from brancharchitect.tree_interpolation.types import TreeInterpolationSequence


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_pair_solution_pivots_follow_lattice_execution_order():
    encoding = {"A": 0, "B": 1, "C": 2, "D": 3}
    child_pivot = Partition((0, 1), encoding)
    root_pivot = Partition((0, 1, 2, 3), encoding)
    child_mover = Partition((0,), encoding)
    root_mover = Partition((2,), encoding)

    sequence = TreeInterpolationSequence(
        interpolated_trees=[object(), object(), object(), object(), object()],
        active_pivot_edges=[None, child_pivot, child_pivot, root_pivot, None],
        current_subtree_highlights=[
            None,
            [child_mover],
            [child_mover],
            [root_mover],
            None,
        ],
        pair_interpolated_tree_counts=[3],
        affected_subtrees_by_split_list=[
            {
                root_pivot: [root_mover],
                child_pivot: [child_mover],
            }
        ],
        attachment_edge_maps=[
            {
                root_pivot: {
                    root_mover: {"source": root_pivot, "destination": root_pivot}
                },
                child_pivot: {
                    child_mover: {"source": child_pivot, "destination": child_pivot}
                },
            }
        ],
        spr_move_events_list=[[]],
    )

    contract = build_temporal_contract(
        sequence,
        robinson_foulds_distances=[1.0],
        weighted_robinson_foulds_distances=[2.0],
    )

    pair_solution = contract["pairs"][0]["solution"]
    assert list(pair_solution["affected_subtrees_by_split"]) == [
        "[0, 1]",
        "[0, 1, 2, 3]",
    ]
    assert list(pair_solution["attachment_edges_by_split"]) == [
        "[0, 1]",
        "[0, 1, 2, 3]",
    ]
    assert [
        event["split"]
        for event in contract["temporal_events"]
        if event["event_type"] == "split_change"
    ] == [[0, 1], [0, 1, 2, 3]]


def test_temporal_contract_pipeline_does_not_export_legacy_intermediate_types():
    legacy_type_names = [
        "".join(["Distance", "Metrics"]),
        "".join(["Tree", "Pair", "Solution"]),
        "".join(["Pair", "Interpolation", "Context"]),
    ]
    legacy_file_names = [
        "_".join(["distance", "metrics"]) + ".py",
        "_".join(["tree", "pair", "solution"]) + ".py",
    ]
    legacy_builder_name = "_".join(["build", "pair", "solutions"])
    duplicate_bridge_name = "_".join(["build", "pair", "contexts"])
    checked_paths = [
        REPO_ROOT / "brancharchitect" / "movie_pipeline",
        REPO_ROOT / "brancharchitect" / "tree_interpolation" / "types",
    ]

    offenders = []
    for checked_path in checked_paths:
        for path in checked_path.rglob("*.py"):
            source = path.read_text(encoding="utf8")
            for legacy_name in legacy_type_names:
                if legacy_name in source:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}: {legacy_name}")
            if legacy_builder_name in source:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {legacy_builder_name}")
            if duplicate_bridge_name in source:
                offenders.append(
                    f"{path.relative_to(REPO_ROOT)}: {duplicate_bridge_name}"
                )

    for checked_path in checked_paths:
        for legacy_file_name in legacy_file_names:
            if list(checked_path.rglob(legacy_file_name)):
                offenders.append(
                    f"{checked_path.relative_to(REPO_ROOT)}: {legacy_file_name}"
                )

    assert offenders == []
