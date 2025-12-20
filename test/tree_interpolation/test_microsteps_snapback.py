import pytest
from brancharchitect.tree import Node
from brancharchitect.tree_interpolation.subtree_paths.execution.microsteps import (
    build_microsteps_for_selection,
)
from brancharchitect.elements.partition import Partition


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

    # Verify initial order is A, C, B
    assert list(reordered.get_current_order()) == ["A", "C", "B"]

    # 2. Define inputs for build_microsteps_for_selection
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
    # Since build_microsteps_for_selection calls other complex functions,
    # we might just test the logic we changed: create_subtree_grafted_tree result.

    # However, we can't easily mock inside the function.
    # Let's rely on the fact that we removed the forced reordering.
    # If we force reordering to ["A", "C", "B"] on ((A, B), C), it should fail or produce crossing.
    # If we don't force it, it should adopt a valid order (e.g. ["A", "B", "C"]).

    try:
        trees, edges, final_tree, subtree_tracker = build_microsteps_for_selection(
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


if __name__ == "__main__":
    test_microsteps_snapback_consistency()
