from unittest.mock import Mock

from brancharchitect.elements.partition import Partition
from brancharchitect.tree_interpolation import sequential_interpolation
from brancharchitect.tree_interpolation.sequential_interpolation import (
    SequentialInterpolationBuilder,
)
from brancharchitect.tree_interpolation.types import TreePairInterpolation


def test_builder_serializes_solution_maps_in_source_destination_order(monkeypatch):
    source_tree = Mock()
    destination_tree = Mock()
    interpolated_tree = Mock()
    source_tree.deep_copy.return_value = source_tree
    destination_tree.deep_copy.return_value = destination_tree

    pivot = Partition((0,), {"A": 0, "B": 1})
    solution = Partition((1,), {"A": 0, "B": 1})
    source_map = {pivot: {solution: Partition((0,), {"A": 0, "B": 1})}}
    destination_map = {pivot: {solution: Partition((1,), {"A": 0, "B": 1})}}

    monkeypatch.setattr(
        sequential_interpolation,
        "process_tree_pair_interpolation",
        lambda *args, **kwargs: TreePairInterpolation(
            trees=[interpolated_tree],
            current_pivot_edge_tracking=[pivot],
            current_subtree_tracking=[[solution]],
            jumping_subtree_solutions={pivot: [solution]},
        ),
    )
    monkeypatch.setattr(
        sequential_interpolation,
        "generate_solution_mappings",
        lambda solutions, source, destination: (source_map, destination_map),
    )

    builder = SequentialInterpolationBuilder()
    result = builder._process_pair(source_tree, destination_tree, 0, None)

    assert result is interpolated_tree
    assert builder.source_mappings == [source_map]
    assert builder.target_mappings == [destination_map]

    sequence = builder._finalize_sequence(original_tree_count=2)
    assert sequence.solution_to_source_maps == [source_map]
    assert sequence.solution_to_destination_maps == [destination_map]
    pair_solutions, _ = sequence.build_pair_solutions([0, 2])
    assert pair_solutions["pair_0_1"]["solution_to_source_map"] == source_map
    assert pair_solutions["pair_0_1"]["solution_to_destination_map"] == destination_map
