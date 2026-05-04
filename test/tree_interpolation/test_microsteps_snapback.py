import pytest
from brancharchitect.tree import Node
import brancharchitect.tree_interpolation.subtree_paths.execution.pivot_edge_interpolation_frame_builder as frame_builder
from brancharchitect.elements.partition import Partition
from brancharchitect.parser.newick_parser import parse_newick


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

    # 2. Define inputs for build_frames_for_subtree
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
    selection = {
        "subtree": pivot,
        "expand": {"path_segment": [split_ab]},
        "collapse": {"path_segment": []},
    }

    # 3. Run microsteps (we need to mock the intermediate functions or rely on them working)
    # Since build_frames_for_subtree calls other complex functions,
    # we might just test the logic we changed: create_subtree_grafted_tree result.

    # However, we can't easily mock inside the function.
    # Let's rely on the fact that we removed the forced reordering.
    # If we force reordering to ["A", "C", "B"] on ((A, B), C), it should fail or produce crossing.
    # If we don't force it, it should adopt a valid order (e.g. ["A", "B", "C"]).

    try:
        trees, edges, final_tree, subtree_tracker = frame_builder.build_frames_for_subtree(
            interpolation_state=reordered,  # Use reordered as start state for simplicity
            destination_tree=dest,
            current_pivot_edge=pivot,
            selection=selection,
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


def test_frame_builder_does_not_mutate_appended_frames(monkeypatch):
    """A frame's order should be final at append time, not rewritten later."""
    source = parse_newick("(A:1,B:1,C:1,D:1);")
    destination = parse_newick("((A:1,B:1):1,C:1,D:1);", encoding=source.taxa_encoding)

    encoding = source.taxa_encoding
    pivot = Partition(tuple(sorted(encoding.values())), encoding)
    mover = Partition((encoding["A"], encoding["C"]), encoding)
    split_ab = Partition((encoding["A"], encoding["B"]), encoding)

    appended_orders = []
    original_append_frame = frame_builder._append_frame

    def record_append(trees, edges, tree, edge, subtree_tracker, partition_group):
        original_append_frame(trees, edges, tree, edge, subtree_tracker, partition_group)
        appended_orders.append(list(trees[-1].get_current_order()))

    monkeypatch.setattr(frame_builder, "_append_frame", record_append)

    trees, _edges, _final_tree, _subtree_tracker = frame_builder.build_frames_for_subtree(
        interpolation_state=source,
        destination_tree=destination,
        current_pivot_edge=pivot,
        selection={
            "subtree": mover,
            "collapse": {"path_segment": []},
            "expand": {"path_segment": [split_ab]},
        },
    )

    final_orders = [list(tree.get_current_order()) for tree in trees]
    assert appended_orders == final_orders


if __name__ == "__main__":
    test_microsteps_snapback_consistency()
