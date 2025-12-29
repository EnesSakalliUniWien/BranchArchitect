
from brancharchitect.tree import Node, ReorderStrategy

def test_reorder_constraint():
    # Construct tree ((A, B), C)
    # Topology: Root -> Child1(A, B), Child2(C)
    child1 = Node(name="AB", children=[Node(name="A"), Node(name="B")])
    child2 = Node(name="C")
    root = Node(name="Root", children=[child1, child2])

    # Initialize encoding
    root.initialize_split_indices({"A": 0, "B": 1, "C": 2})

    print(f"Original Order: {root.get_current_order()}")
    # Should be A, B, C (or similar depending on set iteration, but let's assume A,B,C)

    # CASE 1: Try to insert C between A and B
    # Desired: A, C, B
    target_order = ["A", "C", "B"]
    print(f"\nAttempting Reorder to: {target_order}")

    root.reorder_taxa(target_order, strategy=ReorderStrategy.MINIMUM)

    result = list(root.get_current_order())
    print(f"Resulting Order: {result}")

    if result == ["A", "B", "C"]:
        print(">> C was forced AFTER (A,B) - Snap Effect")
    elif result == ["C", "A", "B"]:
        print(">> C was forced BEFORE (A,B) - Snap Effect")
    elif result == ["A", "C", "B"]:
        print(">> C successfully inserted (Topology Violated? or Polytomy?)")
    else:
        print(">> Other result")

    # CASE 2: Try to move C before A
    # Desired: C, A, B
    target_order_2 = ["C", "A", "B"]
    print(f"\nAttempting Reorder to: {target_order_2}")

    root.reorder_taxa(target_order_2, strategy=ReorderStrategy.MINIMUM)
    result_2 = list(root.get_current_order())
    print(f"Resulting Order: {result_2}")

if __name__ == "__main__":
    test_reorder_constraint()
