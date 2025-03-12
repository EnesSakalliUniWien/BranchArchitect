import pytest
from brancharchitect.tree import Node
from brancharchitect.io import parse_newick
from brancharchitect.plot.tree_plot import plot_circular_trees_in_a_row

from brancharchitect.leaforder.tree_order_optimisation_local import (
    circular_distance_tree_pair,
    optimize_unique_splits,
    optimize_s_edge_splits,
    improve_single_pair_classic,
)


def create_simple_tree(order):
    # Create a simple linear tree for demonstration: A-(B-(C))
    # The Node class should have a constructor allowing name and children
    nodes = [Node(name=name) for name in order]
    # Link them linearly: node[i] as parent of node[i+1]
    for i in range(len(nodes) - 1):
        nodes[i].children.append(nodes[i + 1])
    return nodes[0]  # root is the first node


def test_reorder_subtree_taxa_1():
    # Create a sample tree
    root = Node(name="root")
    child1 = Node(name="child1")
    child2 = Node(name="child2")
    leaf1 = Node(name="leaf1")
    leaf2 = Node(name="leaf2")
    leaf3 = Node(name="leaf3")
    leaf4 = Node(name="leaf4")

    root.children = [child1, child2]
    child1.children = [leaf1, leaf2]
    child2.children = [leaf3, leaf4]

    # Define the new order
    new_order = ["leaf3", "leaf4", "leaf1", "leaf2"]

    # Reorder the subtree taxa
    root.reorder_taxa(new_order)

    # Verify the order of children in the subtree
    assert root.children[0] == child2
    assert root.children[1] == child1
    assert child1.children[0] == leaf1
    assert child1.children[1] == leaf2
    assert child2.children[0] == leaf3
    assert child2.children[1] == leaf4


def test_reorder_subtree_taxa_2():
    # Create a sample tree
    root = Node(name="root")
    leaf3 = Node(name="leaf3")
    leaf4 = Node(name="leaf4")

    root.children = [leaf3, leaf4]

    # Define the new order
    new_order = ["leaf4", "leaf3"]

    # Reorder the subtree taxa
    root.reorder_taxa(new_order)

    # Verify the order of children in the subtree
    assert root.children[0] == leaf4
    assert root.children[1] == leaf3


def test_optimization_with_example_reordering_common_splits_two():
    # Example trees provided
    tree1_newick = "(((B,C),((E,F),(D,A))),(O1,O2));" + "(((B,D),(C,E)),(O1,F),(A,O2));"
    # Parse the trees
    tree1, tree2 = parse_newick(
        tree1_newick, order=["A", "B", "C", "D", "E", "F", "O1", "O2"]
    )
    # display_tree_pair(tree1, tree2)
    # Calculate distance before optimization
    distance_before = circular_distance_tree_pair(tree1, tree2)

    improve_single_pair_classic(
        tree1, tree2, [optimize_s_edge_splits, optimize_unique_splits]
    )

    distance_after = circular_distance_tree_pair(tree1, tree2)
    print(distance_after)
    # Assertions to check if the optimization reduced the distance
    assert distance_after < distance_before, "Optimization did not reduce the distance"


def test_improve_single_pair_on_example_one():
    # Example trees provided
    tree_newick = "(((B,C),((E,F),(D,A))),(O1,O2));" + "(((B,C),A),(D,(E,F)),(O1,O2));"
    tree1, tree2 = parse_newick(
        tree_newick, order=["B", "C", "A", "E", "F", "D", "O1", "O2"]
    )

    distance_before = circular_distance_tree_pair(tree2, tree1)
    # Execute improvement
    print(tree1, tree2)
    improve_single_pair_classic(
        tree1, tree2, [optimize_s_edge_splits, optimize_unique_splits]
    )

    distance_after = circular_distance_tree_pair(tree1, tree2)

    # Verify that distance did not increase
    assert (
        distance_after <= distance_before
    ), "Distance should not increase after optimization."


def test_improve_single_pair_on_example_two():
    tree_newick = "(((B,C),((E,F),(D,A))),(O1,O2));((((B,C),A),(D,(E,F))),(O1,O2));"
    tree1, tree2 = parse_newick(
        tree_newick, order=["B", "C", "A", "E", "F", "D", "O1", "O2"]
    )

    distance_before = circular_distance_tree_pair(tree2, tree1)
    improve_single_pair_classic(
        tree1, tree2, [optimize_s_edge_splits, optimize_unique_splits]
    )
    distance_after = circular_distance_tree_pair(tree1, tree2)

    # Check that distance reduced
    assert (
        distance_after < distance_before
    ), "Distance should reduce after optimization."


def test_improve_single_pair_reverts_on_no_improvement():
    tree_newick = "(A,(B,C));(A,(B,C));"
    tree1, tree2 = parse_newick(tree_newick, order=["A", "B", "C"])

    initial_distance = circular_distance_tree_pair(tree1, tree2)
    original_tree2_order = tree2.get_current_order()

    # Attempt improvements twice in a situation where no improvement is possible.
    improve_single_pair_classic(
        tree1, tree2, [optimize_s_edge_splits, optimize_unique_splits]
    )
    # Second call in reverse order
    improve_single_pair_classic(
        tree2, tree1, [optimize_s_edge_splits, optimize_unique_splits]
    )
    final_distance = circular_distance_tree_pair(tree1, tree2)

    assert final_distance == initial_distance, "Distance should remain the same."
    assert (
        tree2.get_current_order() == original_tree2_order
    ), "Tree2 order should revert to original."


def test_optimize_s_edge_and_unique_splits_on_common_splits_example_one():
    tree_newick = (
        "(((B,C),((E,F),(D,A))),(O1,O2));"
        + "((((B,C),A),(D,(E,F))),(O1,O2));"
        + "(((B,D),(C,E)),(O1,F),(A,O2));"
    )
    trees = parse_newick(tree_newick, order=["A", "B", "C", "D", "E", "F", "O1", "O2"])
    tree1, tree2, _ = trees[0], trees[1], trees[2]

    distance_before = circular_distance_tree_pair(tree1, tree2)
    improve_single_pair_classic(
        tree1, tree2, [optimize_s_edge_splits, optimize_unique_splits]
    )
    distance_after = circular_distance_tree_pair(tree1, tree2)

    # Verify that distance did not increase
    assert (
        distance_after <= distance_before
    ), "Distance should not increase after optimization."


def test_improve_single_pair_with_common_splits_example_two():
    tree_newick = "(((B,C),((E,F),(D,A))),(O1,O2));((((B,C),A),(D,(E,F))),(O1,O2));(((B,D),(C,E)),(O1,F),(A,O2));"
    tree1, tree2, _ = parse_newick(
        tree_newick, order=["A", "B", "C", "D", "E", "F", "O1", "O2"]
    )

    distance_before = circular_distance_tree_pair(tree1, tree2)
    improve_single_pair_classic(
        tree1, tree2, [optimize_s_edge_splits, optimize_unique_splits]
    )
    improve_single_pair_classic(
        tree2, tree1, [optimize_s_edge_splits, optimize_unique_splits]
    )

    distance_after = circular_distance_tree_pair(tree1, tree2)

    assert (
        distance_after <= distance_before
    ), "Distance should not increase after optimization."


@pytest.mark.parametrize(
    "tree_newick",
    [
        "(((B,C),((E,F),(D,A))),(O1,O2));(((B,C),A),(D,(E,F)),(O1,O2));",
    ],
)
def test_can_improve_single_pair_example_one(tmp_path, tree_newick):
    """Example 1: Verify that optimization lowers the circular distance between two trees."""
    # Parse trees
    tree1, tree2 = parse_newick(
        tree_newick, order=["B", "C", "A", "E", "F", "D", "O1", "O2"]
    )

    # Measure distance before optimization
    distance_before = circular_distance_tree_pair(tree1, tree2)

    # Save a plot of the initial trees
    figure_dir = tmp_path / "test_can_improve_single_pair_example_one"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig_before = plot_circular_trees_in_a_row([tree1, tree2])
    with open(figure_dir / "before.svg", "w", encoding="utf-8") as f:
        f.write(fig_before)

    # Optimize
    improve_single_pair_classic(
        tree1, tree2, [optimize_s_edge_splits, optimize_unique_splits]
    )

    # Save a plot of the trees after optimization
    fig_after = plot_circular_trees_in_a_row([tree1, tree2])
    with open(figure_dir / "after.svg", "w", encoding="utf-8") as f:
        f.write(fig_after)

    # Measure distance after optimization
    distance_after = circular_distance_tree_pair(tree1, tree2)

    # Assert that distance has not increased
    assert (
        distance_after <= distance_before
    ), f"Optimization should not make distance worse. Before={distance_before}, After={distance_after}"


@pytest.mark.parametrize(
    "tree_newick",
    ("(((B,C),((E,F),(D,A))),(O1,O2));" + "((((B,C),A),(D,(E,F))),(O1,O2));",),
)
def test_can_improve_single_pair_example_two(tmp_path, tree_newick):
    """Example 2: Verify that optimization reduces or maintains the distance on a second example."""
    tree1, tree2 = parse_newick(
        tree_newick, order=["B", "C", "A", "E", "F", "D", "O1", "O2"]
    )
    distance_before = circular_distance_tree_pair(tree1, tree2)

    figure_dir = tmp_path / "test_can_improve_single_pair_example_two"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig_before = plot_circular_trees_in_a_row([tree1, tree2])
    with open(figure_dir / "before.svg", "w", encoding="utf-8") as f:
        f.write(fig_before)

    improve_single_pair_classic(
        tree1, tree2, [optimize_s_edge_splits, optimize_unique_splits]
    )

    fig_after = plot_circular_trees_in_a_row([tree1, tree2])
    with open(figure_dir / "before.svg", "w", encoding="utf-8") as f:
        f.write(fig_after)

    distance_after = circular_distance_tree_pair(tree1, tree2)
    assert (
        distance_after <= distance_before
    ), f"Distance should reduce or remain equal. Before={distance_before}, After={distance_after}"


def test_no_improvement_reverts_trees(tmp_path):
    """
    Verify that if no improvement is possible, trees revert to their original arrangement.
    """
    tree_newick = "(A,(B,C));(A,(B,C));"
    tree1, tree2 = parse_newick(tree_newick, order=["A", "B", "C"])
    distance_before = circular_distance_tree_pair(tree1, tree2)

    figure_dir = tmp_path / "test_no_improvement_reverts_trees"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig_before = plot_circular_trees_in_a_row([tree1, tree2])
    with open(figure_dir / "before.svg", "w", encoding="utf-8") as f:
        f.write(fig_before)

    # Attempt optimizations
    improve_single_pair_classic(
        tree1, tree2, [optimize_s_edge_splits, optimize_unique_splits]
    )
    improve_single_pair_classic(
        tree1, tree2, [optimize_s_edge_splits, optimize_unique_splits]
    )

    fig_after = plot_circular_trees_in_a_row([tree1, tree2])
    with open(figure_dir / "after.svg", "w", encoding="utf-8") as f:
        f.write(fig_after)

    distance_after = circular_distance_tree_pair(tree1, tree2)
    # No improvement should occur
    assert (
        distance_after == distance_before
    ), f"Distance should remain the same if no improvement is found. Before={distance_before}, After={distance_after}"


def test_s_edge_and_unique_splits_on_common_splits_example_one(tmp_path):
    """Checks s-edge + unique splits on a multiple-tree scenario."""
    tree_newick = (
        "(((B,C),((E,F),(D,A))),(O1,O2));"
        + "((((B,C),A),(D,(E,F))),(O1,O2));"
        + "(((B,D),(C,E)),(O1,F),(A,O2));"
    )

    tree1, tree2, tree3 = parse_newick(
        tree_newick, order=["A", "B", "C", "D", "E", "F", "O1", "O2"]
    )
    distance_before = circular_distance_tree_pair(tree1, tree2)

    figure_dir = tmp_path / "test_s_edge_and_unique_splits_on_common_splits_example_one"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig_before = plot_circular_trees_in_a_row([tree1, tree2])
    with open(figure_dir / "before.svg", "w", encoding="utf-8") as f:
        f.write(fig_before)

    improve_single_pair_classic(
        tree1, tree2, [optimize_s_edge_splits, optimize_unique_splits]
    )

    fig_after = plot_circular_trees_in_a_row([tree1, tree2])
    with open(figure_dir / "after.svg", "w", encoding="utf-8") as f:
        f.write(fig_after)

    distance_after = circular_distance_tree_pair(tree1, tree2)
    assert (
        distance_after <= distance_before
    ), f"Distance should not increase after optimization. Before={distance_before}, After={distance_after}"


def test_improve_single_pair_common_splits_example_two(tmp_path):
    """Further test of s-edge + unique splits on a second multi-tree scenario."""
    tree_newick = (
        "(((B,C),((E,F),(D,A))),(O1,O2));"
        + "((((B,C),A),(D,(E,F))),(O1,O2));"
        + "(((B,D),(C,E)),(O1,F),(A,O2));"
    )
    tree1, tree2, tree3 = parse_newick(
        tree_newick, order=["A", "B", "C", "D", "E", "F", "O1", "O2"]
    )
    distance_before = circular_distance_tree_pair(tree1, tree2)

    figure_dir = tmp_path / "test_improve_single_pair_common_splits_example_two"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig_before = plot_circular_trees_in_a_row([tree1, tree2])
    with open(figure_dir / "before.svg", "w", encoding="utf-8") as f:
        f.write(fig_before)

    improve_single_pair_classic(
        tree1=tree1,
        tree2=tree2,
        rotation_functions=[optimize_s_edge_splits, optimize_unique_splits],
    )

    fig_after = plot_circular_trees_in_a_row([tree1, tree2])
    with open(figure_dir / "after.svg", "w", encoding="utf-8") as f:
        f.write(fig_after)

    distance_after = circular_distance_tree_pair(tree1, tree2)
    assert (
        distance_after <= distance_before
    ), f"Distance should not be higher. Before={distance_before}, After={distance_after}"


def test_improve_single_pair_common_splits_example_three(tmp_path):
    """Verifies behavior on trees with only partial differences, repeated splits scenario."""
    tree_newick = (
        "(((B,C),((E,F),(D,A))),(O1,O2));"
        "(((B,C),((E,F),(D,A))),(O1,O2));"
        "(((B,D),(C,E)),(O1,F),(A,O2));"
    )
    tree1, tree2, tree3 = parse_newick(
        tree_newick, order=["A", "B", "C", "D", "E", "F", "O1", "O2"]
    )
    distance_before = circular_distance_tree_pair(tree1, tree2)

    figure_dir = tmp_path / "test_improve_single_pair_common_splits_example_three"
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig_before = plot_circular_trees_in_a_row([tree1, tree2, tree3])
    with open(figure_dir / "before.svg", "w", encoding="utf-8") as f:
        f.write(fig_before)

    improve_single_pair_classic(
        tree1, tree2, [optimize_s_edge_splits, optimize_s_edge_splits]
    )

    fig_after = plot_circular_trees_in_a_row([tree1, tree2, tree3])
    with open(figure_dir / "after.svg", "w", encoding="utf-8") as f:
        f.write(fig_after)

    distance_after = circular_distance_tree_pair(tree1, tree2)
    assert (
        distance_after <= distance_before
    ), f"Distance should not increase. Before={distance_before}, After={distance_after}"


def test_improve_single_pair_common_splits_example_four(tmp_path):
    """Repeated attempts at improving a pair of trees with common splits."""
    tree_newick = "(((B,C),((E,F),(D,A))),(O1,O2));" + "(((B,D),(C,E)),(O1,F),(A,O2));"
    tree1, tree2 = parse_newick(
        tree_newick, order=["A", "B", "C", "D", "E", "F", "O1", "O2"]
    )
    distance_before = circular_distance_tree_pair(tree1, tree2)

    figure_dir = tmp_path / "test_improve_single_pair_common_splits_example_four"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig_before = plot_circular_trees_in_a_row([tree1, tree2])
    with open(figure_dir / "before.svg", "w", encoding="utf-8") as f:
        f.write(fig_before)

    for _ in range(10):
        improve_single_pair_classic(
            tree1,
            tree2,
            [optimize_unique_splits, optimize_s_edge_splits],
        )
        improve_single_pair_classic(
            tree2,
            tree1,
            [optimize_unique_splits, optimize_s_edge_splits],
        )

    fig_after = plot_circular_trees_in_a_row([tree1, tree2])
    with open(figure_dir / "after.svg", "w", encoding="utf-8") as f:
        f.write(fig_after)

    distance_after = circular_distance_tree_pair(tree1, tree2)
    assert (
        distance_after <= distance_before
    ), f"Distance should not increase. Before={distance_before}, After={distance_after}"


def test_optimize_s_edge_and_unique_splits_on_common_splits_example_five(tmp_path):
    """Verifies that s-edge + unique splits can optimize on a final example."""
    tree_newick = "(((B,C),((E,F),(D,A))),(O1,O2));" "(((B,D),(C,E)),(O1,F),(A,O2));"
    tree1, tree2 = parse_newick(
        tree_newick, order=["A", "B", "C", "D", "E", "F", "O1", "O2"]
    )
    distance_before = circular_distance_tree_pair(tree1, tree2)

    figure_dir = (
        tmp_path
        / "test_optimize_s_edge_and_unique_splits_on_common_splits_example_five"
    )
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig_before = plot_circular_trees_in_a_row([tree1, tree2])
    with open(figure_dir / "before.svg", "w", encoding="utf-8") as f:
        f.write(fig_before)

    optimize_s_edge_splits(
        tree1=tree2,
        tree2=tree1,
        reference_order=tree2.get_current_order(),
        rotated_splits=set(),
    )
    optimize_unique_splits(
        tree1=tree2,
        tree2=tree1,
        reference_order=tree2.get_current_order(),
        rotated_splits=set(),
    )

    distance_after = circular_distance_tree_pair(tree1, tree2)
    fig_after = plot_circular_trees_in_a_row([tree1, tree2])
    with open(figure_dir / "after.svg", "w", encoding="utf-8") as f:
        f.write(fig_after)

    # Optionally verify the distance
    assert distance_after <= distance_before, (
        f"Distance should not be higher after direct s-edge and unique splits. "
        f"Before={distance_before}, After={distance_after}"
    )
