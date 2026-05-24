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


def _unary_internal_node_splits(tree):
    return [
        tuple(node.split_indices.indices)
        for node in tree.traverse()
        if len(node.children) == 1
    ]


def _load_publication_bootstrap_trees(count=10):
    pytest.skip(
        "historical FastTree-specific 24-taxa fixture is not retained in publication_data"
    )


def _load_paper_example_trees():
    data_path = (
        Path(__file__).resolve().parents[4]
        / "publication_data"
        / "figure_example"
        / "paper_example.tree"
    )
    if not data_path.exists():
        pytest.skip("paper example tree data is not available")

    return parse_newick(
        data_path.read_text(),
        force_list=True,
        treat_zero_as_epsilon=True,
    )


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


def test_ostrich_bug_fixture_emits_ostrich_as_separate_mover_step():
    data_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "current_testfiles"
        / "ostrich_bug_example.tree"
    )
    source, destination = parse_newick(data_path.read_text(), force_list=True)[:2]

    result = process_tree_pair_interpolation(source, destination)
    encoding = source.taxa_encoding
    ostrich = encoding["Ostrich"]

    ostrich_events = [
        event
        for event in result.spr_move_events
        if event["driver_subtree"].indices == (ostrich,)
    ]

    assert len(ostrich_events) == 1
    for event in result.spr_move_events:
        highlight_sets = {tuple(group.indices) for group in event["highlight_group"]}
        if (ostrich,) in highlight_sets:
            assert highlight_sets == {(ostrich,)}


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


def test_sequential_interpolation_landing_delimiter_preserves_order_continuity():
    source = parse_newick("((A:1,B:1):1,C:1);")
    destination = parse_newick("(A:1,(B:1,C:1):1);")
    destination.initialize_split_indices(source.taxa_encoding)

    pair_result = process_tree_pair_interpolation(
        source.deep_copy(build_split_index=False),
        destination.deep_copy(build_split_index=False),
    )
    sequence = SequentialInterpolationBuilder().build([source, destination])

    assert pair_result.trees
    assert sequence.pair_interpolated_tree_counts == [len(pair_result.trees) - 1]
    assert len(sequence.interpolated_trees) == len(pair_result.trees) + 1
    assert sequence.get_original_tree_indices() == [
        0,
        len(sequence.interpolated_trees) - 1,
    ]
    _assert_same_topology_and_branch_lengths(
        sequence.interpolated_trees[-1], destination
    )
    assert (
        sequence.interpolated_trees[-2].get_current_order()
        == sequence.interpolated_trees[-1].get_current_order()
    )
    assert sequence.active_pivot_edges[-1] is None
    assert sequence.active_pivot_edges[-2] is not None
    assert sequence.spr_move_events_list[0]
    assert sequence.spr_move_events_list[0][-1]["step_range"][1] == (
        sequence.pair_interpolated_tree_counts[0] - 1
    )


def test_sequential_interpolation_uses_destination_only_as_delimiter():
    source = parse_newick("((A:1,B:1):1,C:1);")
    destination = parse_newick("(A:1,(B:1,C:1):1);")
    destination.initialize_split_indices(source.taxa_encoding)

    pair_result = process_tree_pair_interpolation(
        source.deep_copy(build_split_index=False),
        destination.deep_copy(build_split_index=False),
    )
    sequence = SequentialInterpolationBuilder().build([source, destination])

    assert pair_result.trees
    _assert_same_topology_and_branch_lengths(pair_result.trees[-1], destination)
    assert sequence.pair_interpolated_tree_counts == [len(pair_result.trees) - 1]
    assert len(sequence.interpolated_trees) == len(pair_result.trees) + 1
    assert sequence.get_original_tree_indices() == [
        0,
        len(sequence.interpolated_trees) - 1,
    ]
    _assert_same_topology_and_branch_lengths(
        sequence.interpolated_trees[-1], destination
    )
    assert sequence.active_pivot_edges[-1] is None
    assert sequence.active_pivot_edges[-2] is not None
    assert _weighted_splits(sequence.interpolated_trees[-2]) != _weighted_splits(
        destination
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

    precomputed_solutions = [
        LatticeSolver(trees[index], trees[index + 1]).solve_iteratively()[0]
        for index in range(len(trees) - 1)
    ]
    TreeOrderOptimizer(
        trees,
        precomputed_lattice_solutions=precomputed_solutions,
    ).optimize_with_anchor_ordering(
        anchor_weight_policy="destination",
        circular=True,
        circular_boundary_policy="between_anchor_blocks",
    )

    sequence = SequentialInterpolationBuilder(
        precomputed_lattice_solutions=precomputed_solutions,
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
    ostrich = leaves["Ostrich"]
    great_rhea = leaves["GreatRhea"]
    lesser_rhea = leaves["LesserRhea"]
    cassowary = leaves["Cassowary"]
    emu = leaves["Emu"]
    brown_kiwi = leaves["BrownKiwi"]
    great_spotted_kiwi = leaves["gskiwi"]
    little_spotted_kiwi = leaves["LSKiwi"]
    ec_tinamou = leaves["ECtinamou"]
    g_tinamou = leaves["Gtinamou"]
    crypturellus = leaves["Crypturellus"]

    pair_7 = result["pairs"][7]
    start = pair_7["source_frame_index"]
    end = pair_7["target_frame_index"]
    pair_tracking = result["subtree_highlight_tracking"][start : end + 1]
    pair_highlights = [entry for entry in pair_tracking if entry]

    expected_mover_groups = {
        frozenset([lb_penguin]),
        frozenset([gavia]),
        frozenset([oystercatcher]),
        frozenset([turnstone]),
        frozenset([ostrich]),
        frozenset([great_rhea, lesser_rhea]),
        frozenset([cassowary, emu]),
        frozenset([brown_kiwi, great_spotted_kiwi, little_spotted_kiwi]),
        frozenset([ec_tinamou, g_tinamou, crypturellus]),
    }

    for entry in pair_highlights:
        assert {frozenset(group) for group in entry}.issubset(expected_mover_groups)

    for entry in pair_tracking[1:5]:
        assert {frozenset(group) for group in entry} == {
            frozenset([lb_penguin]),
            frozenset([gavia]),
        }

    oystercatcher_frames = [
        idx
        for idx, entry in enumerate(pair_tracking)
        if entry and any(oystercatcher in group for group in entry)
    ]
    assert oystercatcher_frames
    assert min(oystercatcher_frames) > 4
    for idx in oystercatcher_frames:
        assert {frozenset(group) for group in pair_tracking[idx]} == {
            frozenset([oystercatcher]),
            frozenset([turnstone]),
        }


def test_paper_example_reaches_destination_and_decodes_mover_highlights():
    trees = _load_paper_example_trees()

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
            logger_name="test_paper_example_regression",
        )
    ).process_trees(trees)

    interpolated_trees = result["interpolated_trees"]
    labels_by_index = {
        index: label for label, index in interpolated_trees[0].taxa_encoding.items()
    }
    moving_labels_by_frame = [
        (
            None
            if entry is None
            else [[labels_by_index[index] for index in group] for group in entry]
        )
        for entry in result["subtree_highlight_tracking"]
    ]

    assert len(trees) == 2
    assert len(interpolated_trees) == 13
    assert result["pairs"][0]["source_frame_index"] == 0
    assert result["pairs"][0]["target_frame_index"] == 12
    _assert_same_topology_and_branch_lengths(interpolated_trees[-1], trees[-1])
    assert all(_unary_internal_node_splits(tree) == [] for tree in interpolated_trees)
    destination_clade = interpolated_trees[-1].names_to_partition(
        ("30", "40", "41", "50", "51")
    )
    assert (
        _weighted_splits(interpolated_trees[-1])[_split_key(destination_clade)] == 7.0
    )
    assert moving_labels_by_frame == [
        None,
        [["2"]],
        [["2"]],
        [["2"]],
        [["2"]],
        [["2"]],
        [["1"]],
        [["1"]],
        [["3"]],
        [["3"]],
        [["3"]],
        [["3"]],
        None,
    ]


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
    from brancharchitect.movie_pipeline.tree_interpolation_pipeline import (
        TreeInterpolationPipeline,
    )
    from brancharchitect.movie_pipeline.types import PipelineConfig

    trees = [
        parse_newick("((A:1,B:2):10,(C:3,D:4):20);"),
        parse_newick("(((A:1,B:2):15,C:3):30,D:4);"),
    ]
    trees[1].initialize_split_indices(trees[0].taxa_encoding)

    result = TreeInterpolationPipeline(
        PipelineConfig(enable_rooting=False, use_anchor_ordering=True, circular=True)
    ).process_trees(trees)
    frames = result["frames"]

    input_frames = [frame for frame in frames if frame["frame_type"] == "input_tree"]
    assert input_frames[0]["is_observed_input"] is True
    assert input_frames[-1]["state_semantics"] == "processed_input_tree"
    assert all(
        frame["frame_type"] == "interpolation_frame"
        and frame["is_observed_input"] is False
        for frame in frames[
            input_frames[0]["frame_index"] + 1 : input_frames[-1]["frame_index"]
        ]
    )


def test_frontend_metadata_does_not_emit_duplicate_pivot_tracking():
    pytest.importorskip("flask_compress")
    from webapp.services.trees.frontend_builder import create_empty_movie_data, assemble_frontend_metadata

    metadata = assemble_frontend_metadata(create_empty_movie_data("empty.nwk"))

    assert "pivot_edge_tracking" not in metadata


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
