import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from brancharchitect.parser import parse_newick
from brancharchitect.tree_interpolation.pair_interpolation import (
    process_tree_pair_interpolation,
)
from brancharchitect.tree_interpolation.sequential_interpolation import (
    SequentialInterpolationBuilder,
)


def _split_key(partition):
    return tuple(partition.indices)


def _split_keys(tree):
    return {_split_key(partition) for partition in tree.to_splits()}


def _weighted_splits(tree):
    return {
        _split_key(partition): float(weight)
        for partition, weight in tree.to_weighted_splits().items()
    }


def _assert_same_topology_and_branch_lengths(actual, expected):
    assert _split_keys(actual) == _split_keys(expected)
    assert _weighted_splits(actual) == _weighted_splits(expected)


def _load_publication_bootstrap_trees(count=10):
    data_path = (
        Path(__file__).resolve().parents[4]
        / "publication_data"
        / "bootstrap_example"
        / "24"
        / "all_trees_24.nwk"
    )
    if not data_path.exists():
        pytest.skip("publication bootstrap 24-taxa data is not available")

    newicks = data_path.read_text().strip().splitlines()[:count]
    return parse_newick("\n".join(newicks), force_list=True)


def _event_for_local_step(events, local_step):
    for event in events:
        start, end = event["step_range"]
        if start <= local_step <= end:
            return event
    raise AssertionError(f"No event covers local step {local_step}")


def test_pair_interpolation_final_frame_matches_destination_without_order_snap():
    source = parse_newick("((A:1,B:1):1,(C:1,D:1):1);")
    destination = parse_newick("((A:1,C:1):2,(B:1,D:1):3);")
    destination.initialize_split_indices(source.taxa_encoding)

    result = process_tree_pair_interpolation(source, destination)

    assert len(result.trees) >= 2
    _assert_same_topology_and_branch_lengths(result.trees[-1], destination)
    assert result.trees[-2].get_current_order() == result.trees[-1].get_current_order()


def test_pair_interpolation_final_frame_preserves_destination_semantics():
    source = parse_newick("((A:1,B:2):10,(C:3,D:4):20);")
    destination = parse_newick("(((A:1,B:2):15,C:3):30,D:4);")
    destination.initialize_split_indices(source.taxa_encoding)

    result = process_tree_pair_interpolation(source, destination)

    assert result.trees
    _assert_same_topology_and_branch_lengths(result.trees[-1], destination)


def test_sequential_delimiters_are_observed_topology_and_weight_states():
    trees = [
        parse_newick("((A:1,B:2):10,(C:3,D:4):20);"),
        parse_newick("(((A:1,B:2):15,C:3):30,D:4);"),
        parse_newick("((A:1,C:3):30,(B:2,D:4):40);"),
    ]
    for tree in trees[1:]:
        tree.initialize_split_indices(trees[0].taxa_encoding)

    result = SequentialInterpolationBuilder().build(trees)
    delimiter_indices = result.get_original_tree_indices()

    assert len(delimiter_indices) == len(trees)
    for delimiter_index, input_tree in zip(delimiter_indices, trees):
        _assert_same_topology_and_branch_lengths(
            result.interpolated_trees[delimiter_index], input_tree
        )


def test_sequential_interpolation_keeps_generated_landing_frame_before_anchor():
    source = parse_newick("((A:1,B:1):1,C:1);")
    destination = parse_newick("(A:1,(B:1,C:1):1);")
    destination.initialize_split_indices(source.taxa_encoding)

    pair_result = process_tree_pair_interpolation(
        source.deep_copy(build_split_index=False),
        destination.deep_copy(build_split_index=False),
    )
    sequence = SequentialInterpolationBuilder().build([source, destination])

    assert pair_result.trees
    assert sequence.pair_interpolated_tree_counts == [len(pair_result.trees)]
    assert len(sequence.interpolated_trees) == len(pair_result.trees) + 2
    assert sequence.get_original_tree_indices() == [
        0,
        len(sequence.interpolated_trees) - 1,
    ]
    _assert_same_topology_and_branch_lengths(
        sequence.interpolated_trees[-2], destination
    )
    _assert_same_topology_and_branch_lengths(
        sequence.interpolated_trees[-1], destination
    )
    assert (
        sequence.interpolated_trees[-2].get_current_order()
        == sequence.interpolated_trees[-1].get_current_order()
    )
    assert sequence.current_pivot_edge_tracking[-1] is None
    assert sequence.current_pivot_edge_tracking[-2] is not None
    assert sequence.spr_move_events_list[0]
    assert sequence.spr_move_events_list[0][-1]["step_range"][1] == (
        sequence.pair_interpolated_tree_counts[0] - 1
    )


def test_bootstrap_collapse_frames_do_not_reorder_before_expansion():
    trees = _load_publication_bootstrap_trees()
    sequence = SequentialInterpolationBuilder().build(trees)
    original_indices = sequence.get_original_tree_indices()

    # Pair 4 (rep_9 -> rep_53) and pair 7 (rep_33 -> rep_23) both used to
    # reorder Alligator/Caiman vs tinamous during the collapse frame itself.
    for pair_index in (4, 7):
        pair_start = original_indices[pair_index]
        before_collapse = sequence.interpolated_trees[pair_start + 1 + 4]
        collapsed = sequence.interpolated_trees[pair_start + 1 + 5]

        assert _split_keys(before_collapse) != _split_keys(collapsed)
        assert before_collapse.get_current_order() == collapsed.get_current_order()


def test_bootstrap_topology_frames_do_not_also_reorder_leaves():
    trees = _load_publication_bootstrap_trees()
    sequence = SequentialInterpolationBuilder().build(trees)

    topology_and_order_changes = []
    for index, (source, target) in enumerate(
        zip(sequence.interpolated_trees, sequence.interpolated_trees[1:])
    ):
        if _split_keys(source) != _split_keys(target) and (
            source.get_current_order() != target.get_current_order()
        ):
            topology_and_order_changes.append(index)

    assert topology_and_order_changes == []


def test_bootstrap_circular_pipeline_keeps_topology_frames_order_stable():
    trees = _load_publication_bootstrap_trees()

    from brancharchitect.jumping_taxa.lattice.solvers.lattice_solver import (
        LatticeSolver,
    )
    from brancharchitect.leaforder.tree_order_optimiser import TreeOrderOptimizer

    pair_solutions = [
        LatticeSolver(trees[index], trees[index + 1]).solve_iteratively()[0]
        for index in range(len(trees) - 1)
    ]
    TreeOrderOptimizer(
        trees,
        precomputed_pair_solutions=pair_solutions,
    ).optimize_with_anchor_ordering(
        anchor_weight_policy="destination",
        circular=True,
        circular_boundary_policy="between_anchor_blocks",
    )

    sequence = SequentialInterpolationBuilder(
        precomputed_pair_solutions=pair_solutions,
    ).build(trees)

    interpolated_trees = sequence.interpolated_trees
    topology_and_order_changes = []
    for index, (source, target) in enumerate(
        zip(interpolated_trees, interpolated_trees[1:])
    ):
        if _split_keys(source) != _split_keys(target) and (
            source.get_current_order() != target.get_current_order()
        ):
            topology_and_order_changes.append(index)

    assert topology_and_order_changes == []


def test_bootstrap_pair_7_8_highlights_movers_not_passive_context():
    trees = _load_publication_bootstrap_trees()

    from brancharchitect.movie_pipeline.tree_interpolation_pipeline import (
        TreeInterpolationPipeline,
    )
    from brancharchitect.movie_pipeline.types import PipelineConfig

    result = TreeInterpolationPipeline(
        PipelineConfig(
            enable_rooting=False,
            use_anchor_ordering=True,
            anchor_weight_policy="destination",
            circular=True,
            logger_name="test_pair_7_8_mover_highlights",
        )
    ).process_trees(trees)

    leaves = {
        name: index
        for name, index in result["interpolated_trees"][0].taxa_encoding.items()
    }
    oystercatcher = leaves["oystercatcher"]
    lb_penguin = leaves["LBPenguin"]
    gavia = leaves["GaviaStellata"]
    turnstone = leaves["turnstone"]

    start, end = result["pair_interpolation_ranges"][7]
    pair_tracking = result["subtree_highlight_tracking"][start : end + 1]
    pair_highlights = [entry for entry in pair_tracking if entry]

    for entry in pair_tracking[1:5]:
        assert entry == [[gavia]]

    oystercatcher_frames = [
        idx
        for idx, entry in enumerate(pair_tracking)
        if entry and any(oystercatcher in group for group in entry)
    ]
    assert oystercatcher_frames
    assert min(oystercatcher_frames) > 4
    for idx in oystercatcher_frames:
        assert pair_tracking[idx] == [[oystercatcher]]

    passive_context_taxa = (lb_penguin, turnstone)
    for taxon in passive_context_taxa:
        assert not any(
            any(taxon in group for group in entry) for entry in pair_highlights
        )


def test_identical_topology_pair_aligns_delimiter_order_to_source():
    source = parse_newick("((A:1,B:1):1,(C:1,D:1):1);")
    destination = parse_newick("((C:1,D:1):2,(A:1,B:1):3);")
    destination.initialize_split_indices(source.taxa_encoding)

    sequence = SequentialInterpolationBuilder().build([source, destination])

    assert sequence.pair_interpolated_tree_counts == [0]
    assert len(sequence.interpolated_trees) == 2
    _assert_same_topology_and_branch_lengths(
        sequence.interpolated_trees[-1], destination
    )
    assert (
        sequence.interpolated_trees[0].get_current_order()
        == sequence.interpolated_trees[-1].get_current_order()
    )


def test_sequence_metadata_marks_input_frames_as_observed_tree_states():
    trees = [
        parse_newick("((A:1,B:2):10,(C:3,D:4):20);"),
        parse_newick("(((A:1,B:2):15,C:3):30,D:4);"),
    ]
    trees[1].initialize_split_indices(trees[0].taxa_encoding)

    result = SequentialInterpolationBuilder().build(trees)
    pair_solutions, pair_ranges = result.build_pair_solutions(
        result.get_original_tree_indices()
    )
    result.tree_pair_solutions = pair_solutions
    result.pair_interpolation_ranges = pair_ranges

    sys.modules.setdefault("orjson", types.SimpleNamespace(dumps=lambda _value: b""))
    from brancharchitect.movie_pipeline.tree_interpolation_pipeline import (
        TreeInterpolationPipeline,
    )

    metadata = TreeInterpolationPipeline()._create_global_tree_metadata(
        result.current_pivot_edge_tracking,
        result.get_original_tree_indices(),
    )

    original_indices = result.get_original_tree_indices()
    assert metadata[original_indices[0]]["frame_type"] == "input_tree"
    assert metadata[original_indices[0]]["is_observed_input"] is True
    assert metadata[original_indices[-1]]["state_semantics"] == "processed_input_tree"
    assert all(
        metadata[idx]["frame_type"] == "interpolation_frame"
        and metadata[idx]["is_observed_input"] is False
        for idx in range(original_indices[0] + 1, original_indices[-1])
    )


def test_frontend_pivot_tracking_does_not_mark_input_tree_endpoints():
    pytest.importorskip("flask_compress")
    from webapp.services.trees.frontend_builder import (
        _derive_pivot_edge_tracking_from_events,
    )

    metadata = [
        {
            "tree_pair_key": None,
            "step_in_pair": None,
            "source_tree_global_index": None,
            "frame_type": "input_tree",
            "state_semantics": "processed_input_tree",
            "is_observed_input": True,
        },
        {
            "tree_pair_key": "pair_0_1",
            "step_in_pair": 1,
            "source_tree_global_index": 0,
            "frame_type": "interpolation_frame",
            "state_semantics": "algorithmic_intermediate",
            "is_observed_input": False,
        },
        {
            "tree_pair_key": None,
            "step_in_pair": None,
            "source_tree_global_index": None,
            "frame_type": "input_tree",
            "state_semantics": "processed_input_tree",
            "is_observed_input": True,
        },
    ]

    tracking = _derive_pivot_edge_tracking_from_events(
        metadata,
        {"pair_0_1": [{"step_range": [0, 1], "split": [1, 2]}]},
    )

    assert tracking == [None, [1, 2], None]


def test_anchor_fallback_order_is_independent_of_python_hash_seed():
    script = """
from brancharchitect.parser import parse_newick
from brancharchitect.elements.partition import Partition
from brancharchitect.elements.partition_set import PartitionSet
from brancharchitect.leaforder.anchor_order import blocked_order_and_apply

order = ["A", "B", "C", "D", "E", "F"]
encoding = {name: i for i, name in enumerate(order)}
source = parse_newick("(A:1,B:1,C:1,D:1,E:1,F:1):0;", order=order, encoding=encoding)
destination = parse_newick("(A:2,B:2,C:2,D:2,E:2,F:2):0;", order=order, encoding=encoding)
root = Partition(tuple(range(6)), encoding)
blocked_order_and_apply(
    root,
    {},
    {},
    source,
    destination,
    anchor_weight_policy="destination",
    common_splits=PartitionSet(encoding=encoding),
)
print("|".join(source.get_current_order()))
"""

    outputs = set()
    for seed in ("0", "1", "2", "3"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed
        env["PYTHONPATH"] = os.getcwd()
        output = subprocess.check_output(
            [sys.executable, "-c", script],
            cwd=os.getcwd(),
            env=env,
            text=True,
        ).strip()
        outputs.add(output)

    assert outputs == {"A|B|C|D|E|F"}
