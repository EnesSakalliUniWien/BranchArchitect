from collections import Counter
import inspect

import pytest
from brancharchitect.tree import Node
from brancharchitect.tree_interpolation.subtree_paths.execution.phases import (
    expand_phase,
    subtree_microsteps,
)
from brancharchitect.tree_interpolation.subtree_paths.execution.frames.frame_batch import (
    FrameBatch,
)
from brancharchitect.elements.partition import Partition
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.tree_interpolation.subtree_paths.planning import PivotTransitionStep


def _transition_step(
    subtree: Partition,
    collapse_path: list[Partition] | None = None,
    expand_path: list[Partition] | None = None,
) -> PivotTransitionStep:
    return PivotTransitionStep(
        subtree=subtree,
        collapse_path=tuple(collapse_path or []),
        expand_path=tuple(expand_path or []),
    )


def test_subtree_microsteps_api_uses_current_transition_step_inputs():
    parameters = inspect.signature(
        subtree_microsteps.build_subtree_interpolation_frames
    ).parameters

    assert "selection" in parameters
    assert parameters["selection"].annotation == "PivotTransitionStep"
    assert "is_first_mover" in parameters


def test_microsteps_snapback_consistency():
    """
    Test that microsteps do not introduce inconsistent ordering (snapback)
    when grafting new splits that conflict with the base order.
    """
    # 1. Setup a scenario where 'reordered' has a flat topology and specific order
    #    but 'expand_path' introduces a split that requires a DIFFERENT order.

    # Taxa: A, B, C
    # Reordered (Collapsed): Star topology (A, B, C). Order: A, C, B
    # Expand Path: Split ((A, B), C). Requires A and B to be adjacent.

    # Create base tree (reordered)
    reordered = Node()
    reordered.taxa_encoding = {"A": 0, "B": 1, "C": 2}
    # Create star topology
    reordered.children = [
        Node(name="A", length=0.1),
        Node(name="C", length=0.1),
        Node(name="B", length=0.1),
    ]
    reordered.initialize_split_indices(reordered.taxa_encoding)

    # Verify initial order is A, C, B
    assert list(reordered.get_current_order()) == ["A", "C", "B"]

    # 2. Define inputs for build_subtree_interpolation_frames
    # We need to mock the inputs since we are testing the logic flow

    # Destination tree (has the split)
    dest = Node()
    dest.taxa_encoding = reordered.taxa_encoding
    # ((A, B), C)
    ab_clade = Node()
    ab_clade.children = [Node(name="A", length=0.1), Node(name="B", length=0.1)]
    dest.children = [ab_clade, Node(name="C", length=0.1)]  # Pivot edge (dummy)
    pivot = Partition((0, 1, 2), reordered.taxa_encoding)  # Root

    # Selection with expand path
    # Split {A, B} is (0, 1)
    split_ab = Partition((0, 1), reordered.taxa_encoding)
    selection = _transition_step(pivot, expand_path=[split_ab])

    # 3. Run microsteps (we need to mock the intermediate functions or rely on them working)
    # Since build_subtree_interpolation_frames calls other complex functions,
    # we might just test the logic we changed: create_subtree_grafted_tree result.

    # However, we can't easily mock inside the function.
    # Let's rely on the fact that we removed the forced reordering.
    # If we force reordering to ["A", "C", "B"] on ((A, B), C), it should fail or produce crossing.
    # If we don't force it, it should adopt a valid order (e.g. ["A", "B", "C"]).

    try:
        trees, edges, final_tree, subtree_tracker = (
            subtree_microsteps.build_subtree_interpolation_frames(
                interpolation_state=reordered,  # Use reordered as start state for simplicity
                destination_tree=dest,
                current_pivot_edge=pivot,
                selection=selection,
            )
        )

        # 4. Check the order of the final tree
        final_order = list(final_tree.get_current_order())
        print(f"Final Order: {final_order}")

        # The order MUST respect the split (A, B). A and B must be adjacent.
        # "A", "C", "B" puts C in between -> Invalid.
        a_idx = final_order.index("A")
        b_idx = final_order.index("B")
        c_idx = final_order.index("C")

        # Check adjacency of A and B
        assert abs(a_idx - b_idx) == 1, f"A and B should be adjacent in {final_order}"

    except Exception as e:
        pytest.fail(f"Microsteps failed: {e}")


def test_subtree_microsteps_do_not_mutate_appended_frames(monkeypatch):
    """A frame's order should be final at append time, not rewritten later."""
    source = parse_newick("(A:1,B:1,C:1,D:1);")
    destination = parse_newick("((A:1,B:1):1,C:1,D:1);", encoding=source.taxa_encoding)

    encoding = source.taxa_encoding
    pivot = Partition(tuple(sorted(encoding.values())), encoding)
    mover = Partition((encoding["A"], encoding["C"]), encoding)
    split_ab = Partition((encoding["A"], encoding["B"]), encoding)

    appended_orders = []
    original_append_frame = FrameBatch.append

    def record_append(self, tree, edge, partition_group):
        original_append_frame(self, tree, edge, partition_group)
        appended_orders.append(list(self.trees[-1].get_current_order()))

    monkeypatch.setattr(FrameBatch, "append", record_append)

    trees, _edges, _final_tree, _subtree_tracker = (
        subtree_microsteps.build_subtree_interpolation_frames(
            interpolation_state=source,
            destination_tree=destination,
            current_pivot_edge=pivot,
            selection=_transition_step(mover, expand_path=[split_ab]),
        )
    )

    final_orders = [list(tree.get_current_order()) for tree in trees]
    assert appended_orders == final_orders


def test_joint_expand_group_is_not_treated_as_stable_anchor(monkeypatch):
    """Inactive sibling movers in the same expand group should not anchor graft alignment."""
    source = parse_newick("(B:1,C:1,A:1,D:1);")
    destination = parse_newick("(C:1,(A:1,B:1):1,D:1);", encoding=source.taxa_encoding)

    encoding = source.taxa_encoding
    pivot = Partition(tuple(sorted(encoding.values())), encoding)
    mover_a = Partition((encoding["A"],), encoding)
    mover_b = Partition((encoding["B"],), encoding)
    split_ab = Partition((encoding["A"], encoding["B"]), encoding)
    moving_taxa_calls = []
    original_align_to_source_order = expand_phase.align_to_source_order

    def record_align_to_source_order(tree, source_order, moving_taxa=None):
        moving_taxa_calls.append(set(moving_taxa or set()))
        return original_align_to_source_order(tree, source_order, moving_taxa)

    monkeypatch.setattr(
        expand_phase, "align_to_source_order", record_align_to_source_order
    )

    trees, _edges, final_tree, subtree_tracker = (
        subtree_microsteps.build_subtree_interpolation_frames(
            interpolation_state=source,
            destination_tree=destination,
            current_pivot_edge=pivot,
            selection=_transition_step(mover_a, expand_path=[split_ab]),
            all_mover_partitions=[mover_a, mover_b],
            expand_sibling_groups={
                mover_a: [mover_a, mover_b],
                mover_b: [mover_a, mover_b],
            },
        )
    )

    assert moving_taxa_calls == [{"A", "B"}]
    assert all(
        [partition.taxa for partition in group] == [mover_a.taxa, mover_b.taxa]
        for group in subtree_tracker
    )
    assert list(trees[-1].get_current_order()) == list(final_tree.get_current_order())


def test_reorder_snap_reuses_owned_working_trees_without_extra_snapshots(monkeypatch):
    """Owned working trees can be handed to pending frames without extra copies."""
    source = parse_newick("(A:1,B:1,C:1,D:1);")
    destination = parse_newick("(A:1,C:1,B:1,D:1);", encoding=source.taxa_encoding)
    destination.reorder_taxa(["A", "C", "B", "D"])

    encoding = source.taxa_encoding
    pivot = Partition(tuple(sorted(encoding.values())), encoding)
    mover = Partition((encoding["B"],), encoding)

    copy_counts = Counter()
    original_deep_copy = Node.deep_copy

    def counted_deep_copy(self, *args, **kwargs):
        copy_counts[kwargs.get("build_split_index", True)] += 1
        return original_deep_copy(self, *args, **kwargs)

    monkeypatch.setattr(Node, "deep_copy", counted_deep_copy)

    trees, _edges, _final_tree, _subtree_tracker = (
        subtree_microsteps.build_subtree_interpolation_frames(
            interpolation_state=source,
            destination_tree=destination,
            current_pivot_edge=pivot,
            selection=_transition_step(mover),
        )
    )

    assert [list(tree.get_current_order()) for tree in trees] == [
        ["A", "B", "C", "D"],
        ["A", "C", "B", "D"],
        ["A", "C", "B", "D"],
    ]
    assert copy_counts[True] == 2
    assert copy_counts[False] == 2


def test_collapse_reorder_reuses_owned_working_tree_without_extra_snapshot(monkeypatch):
    """Collapse frames own snapshots, so the collapsed working tree can be reordered."""
    source = parse_newick("((A:1,B:1):1,C:1,D:1);")
    destination = parse_newick("(C:1,A:1,B:1,D:1);", encoding=source.taxa_encoding)
    destination.reorder_taxa(["C", "A", "B", "D"])

    encoding = source.taxa_encoding
    pivot = Partition(tuple(sorted(encoding.values())), encoding)
    mover = Partition((encoding["C"],), encoding)
    split_ab = Partition((encoding["A"], encoding["B"]), encoding)

    copy_count = 0
    original_deep_copy = Node.deep_copy

    def counted_deep_copy(self, *args, **kwargs):
        nonlocal copy_count
        copy_count += 1
        return original_deep_copy(self, *args, **kwargs)

    monkeypatch.setattr(Node, "deep_copy", counted_deep_copy)

    trees, _edges, _final_tree, _subtree_tracker = (
        subtree_microsteps.build_subtree_interpolation_frames(
            interpolation_state=source,
            destination_tree=destination,
            current_pivot_edge=pivot,
            selection=_transition_step(mover, collapse_path=[split_ab]),
            all_mover_partitions=[mover],
        )
    )

    assert [list(tree.get_current_order()) for tree in trees] == [
        ["A", "B", "C", "D"],
        ["A", "B", "C", "D"],
        ["C", "A", "B", "D"],
        ["C", "A", "B", "D"],
    ]
    assert copy_count == 5


def test_expand_snap_avoids_indexed_copy_for_snap_source(monkeypatch):
    """The zero-weight expand frame is snapshotted before mutating the owned tree."""
    source = parse_newick("(A:1,B:1,C:1,D:1);")
    destination = parse_newick("((A:2,B:2):2,C:2,D:2);", encoding=source.taxa_encoding)

    encoding = source.taxa_encoding
    pivot = Partition(tuple(sorted(encoding.values())), encoding)
    mover = Partition((encoding["A"], encoding["C"]), encoding)
    split_ab = Partition((encoding["A"], encoding["B"]), encoding)

    copy_counts = Counter()
    original_deep_copy = Node.deep_copy

    def counted_deep_copy(self, *args, **kwargs):
        copy_counts[kwargs.get("build_split_index", True)] += 1
        return original_deep_copy(self, *args, **kwargs)

    monkeypatch.setattr(Node, "deep_copy", counted_deep_copy)

    trees, _edges, final_tree, _subtree_tracker = (
        subtree_microsteps.build_subtree_interpolation_frames(
            interpolation_state=source,
            destination_tree=destination,
            current_pivot_edge=pivot,
            selection=_transition_step(mover, expand_path=[split_ab]),
        )
    )

    assert trees[-2] is not final_tree
    assert trees[-1] is not final_tree
    assert copy_counts[True] == 1
    assert copy_counts[False] == 5


def test_expand_grafts_private_working_tree_without_copy(monkeypatch):
    """If no pending frame aliases the working tree, grafting can mutate it."""
    source = parse_newick("(A:1,B:1,C:1,D:1);")
    destination = parse_newick("((A:2,B:2):2,C:2,D:2);", encoding=source.taxa_encoding)

    encoding = source.taxa_encoding
    pivot = Partition(tuple(sorted(encoding.values())), encoding)
    split_ab = Partition((encoding["A"], encoding["B"]), encoding)

    copy_counts = Counter()
    original_deep_copy = Node.deep_copy

    def counted_deep_copy(self, *args, **kwargs):
        copy_counts[kwargs.get("build_split_index", True)] += 1
        return original_deep_copy(self, *args, **kwargs)

    monkeypatch.setattr(Node, "deep_copy", counted_deep_copy)

    trees, _edges, final_tree, _subtree_tracker = (
        subtree_microsteps.build_subtree_interpolation_frames(
            interpolation_state=source,
            destination_tree=destination,
            current_pivot_edge=pivot,
            selection=_transition_step(split_ab, expand_path=[split_ab]),
        )
    )

    assert trees[-2] is not final_tree
    assert trees[-1] is not final_tree
    assert copy_counts[True] == 1
    assert copy_counts[False] == 2


def test_early_return_final_state_does_not_alias_appended_frames():
    """The returned state may be mutated by callers without rewriting frames."""
    source = parse_newick("(A:1,B:1,C:1,D:1);")
    destination = parse_newick("(A:1,C:1,B:1,D:1);", encoding=source.taxa_encoding)
    destination.reorder_taxa(["A", "C", "B", "D"])

    encoding = source.taxa_encoding
    pivot = Partition(tuple(sorted(encoding.values())), encoding)
    mover = Partition((encoding["B"],), encoding)

    trees, _edges, final_tree, _subtree_tracker = (
        subtree_microsteps.build_subtree_interpolation_frames(
            interpolation_state=source,
            destination_tree=destination,
            current_pivot_edge=pivot,
            selection=_transition_step(mover),
            is_first_mover=False,
        )
    )

    frame_orders_before = [list(tree.get_current_order()) for tree in trees]
    final_tree.reorder_taxa(["A", "B", "C", "D"])

    assert [list(tree.get_current_order()) for tree in trees] == frame_orders_before


def test_active_split_sequence_uses_deep_copy_index_without_rebuild(monkeypatch):
    """Current-base tree copies should not rebuild split indexes after copying."""
    from brancharchitect.tree_interpolation.subtree_paths.execution.sequence.active_split_sequence import (
        execute_active_split_transition_sequence,
    )

    source = parse_newick("(A:1,B:1,C:1,D:1);")
    destination = parse_newick("(A:1,C:1,B:1,D:1);", encoding=source.taxa_encoding)
    destination.reorder_taxa(["A", "C", "B", "D"])

    encoding = source.taxa_encoding
    pivot = Partition(tuple(sorted(encoding.values())), encoding)
    mover = Partition((encoding["B"],), encoding)

    source.build_split_index()
    destination.build_split_index()

    def fail_rebuild(self):
        raise AssertionError("deep-copied current base tree should already be indexed")

    monkeypatch.setattr(Node, "build_split_index", fail_rebuild)

    trees, edges, subtree_highlights, _events = (
        execute_active_split_transition_sequence(
            source_tree=source,
            destination_tree=destination,
            target_pivot_edges=[pivot],
            jumping_subtree_solutions={pivot: [mover]},
        )
    )

    assert len(trees) == len(edges) == len(subtree_highlights)


def test_active_split_sequence_hands_off_owned_state_without_copying(monkeypatch):
    """The state returned by one pivot is private chaining state for the next."""
    import brancharchitect.tree_interpolation.subtree_paths.execution.sequence.active_split_sequence as active_split_sequence

    source = parse_newick("(A:1,B:1);")
    destination = parse_newick("(A:1,B:1);", encoding=source.taxa_encoding)

    encoding = source.taxa_encoding
    pivot = Partition(tuple(sorted(encoding.values())), encoding)
    mover = Partition((encoding["A"],), encoding)

    monkeypatch.setattr(
        active_split_sequence,
        "build_pivot_subtree_transition_paths",
        lambda *args, **kwargs: ({pivot: {}}, {pivot: {}}),
    )

    seen_base_trees = []
    returned_states = []

    def fake_execute_pivot_edge_interpolation(
        current_base_tree,
        destination_tree,
        source_tree,
        current_pivot_edge,
        collapse_paths_for_pivot_edge,
        expand_paths_for_pivot_edge,
        source_parent_map,
        dest_parent_map,
    ):
        seen_base_trees.append(current_base_tree)
        frame_tree = current_base_tree.deep_copy()
        new_state = current_base_tree.deep_copy()
        returned_states.append(new_state)
        return [frame_tree], [current_pivot_edge], new_state, [[mover]], []

    monkeypatch.setattr(
        active_split_sequence,
        "execute_pivot_edge_interpolation",
        fake_execute_pivot_edge_interpolation,
    )

    active_split_sequence.execute_active_split_transition_sequence(
        source_tree=source,
        destination_tree=destination,
        target_pivot_edges=[pivot, pivot],
        jumping_subtree_solutions={pivot: [mover]},
    )

    assert seen_base_trees[1] is returned_states[0]


def test_align_to_source_order_does_not_repeatedly_collect_leaves(monkeypatch):
    """Alignment ordering should derive sort keys in one traversal."""
    from brancharchitect.tree_interpolation.subtree_paths.execution.layout.tree_order_alignment import (
        align_to_source_order,
    )

    tree = parse_newick("(A:1,(B:1,C:1):1,D:1);")
    original_get_leaves = Node.get_leaves

    def fail_get_leaves(self):
        raise AssertionError("align_to_source_order should not call get_leaves")

    monkeypatch.setattr(Node, "get_leaves", fail_get_leaves)

    align_to_source_order(tree, ["D", "C", "B", "A"], moving_taxa={"B"})

    monkeypatch.setattr(Node, "get_leaves", original_get_leaves)
    assert list(tree.get_current_order()) == ["D", "C", "B", "A"]


if __name__ == "__main__":
    test_microsteps_snapback_consistency()
