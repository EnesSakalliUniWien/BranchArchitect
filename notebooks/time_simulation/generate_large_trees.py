#!/usr/bin/env python3
"""
Script to generate large trees with high number of leaf labels (300) and recombination events.
Creates trees compatible with BranchArchitect's Node structure.
"""

import random
from typing import List
from pathlib import Path

# Import BranchArchitect modules
from brancharchitect.tree import Node
from brancharchitect.io.newick import write_newick


def generate_taxa_names(num_taxa: int = 300) -> List[str]:
    """Generate unique taxa names for the trees."""
    taxa = []

    # Generate names like T001, T002, ..., T300
    for i in range(1, num_taxa + 1):
        taxa.append(f"T{i:03d}")

    return taxa


def create_random_binary_tree(taxa: List[str], add_branch_lengths: bool = True) -> Node:
    """
    Create a random binary tree with the given taxa as leaves.

    Args:
        taxa: List of taxa names to use as leaves
        add_branch_lengths: Whether to add random branch lengths

    Returns:
        Root node of the generated tree
    """
    # Start with all taxa as individual nodes
    nodes = [Node(name=taxon) for taxon in taxa]

    # Randomly join nodes until we have a single root
    while len(nodes) > 1:
        # Randomly select two nodes to join
        idx1 = random.randint(0, len(nodes) - 1)
        node1 = nodes.pop(idx1)

        idx2 = random.randint(0, len(nodes) - 1)
        node2 = nodes.pop(idx2)

        # Create internal node
        parent = Node()
        parent.add_child(node1)
        parent.add_child(node2)

        # Add random branch lengths if requested
        if add_branch_lengths:
            node1.branch_length = random.uniform(0.01, 0.5)
            node2.branch_length = random.uniform(0.01, 0.5)

        nodes.append(parent)

    return nodes[0]


def simulate_recombination_event(
    tree1: Node, tree2: Node, recombination_rate: float = 0.1
) -> Node:
    """
    Simulate a recombination event between two trees.
    Creates a new tree that combines subtrees from both parents.

    Args:
        tree1: First parent tree
        tree2: Second parent tree
        recombination_rate: Probability of recombination at each internal node

    Returns:
        New tree resulting from recombination
    """
    # Get all internal nodes from both trees
    internal_nodes1 = [node for node in tree1.traverse() if node.children]
    internal_nodes2 = [node for node in tree2.traverse() if node.children]

    if not internal_nodes1 or not internal_nodes2:
        return tree1.deep_copy()

    # Start with a copy of tree1
    result = tree1.deep_copy()

    # For each internal node, decide whether to recombine
    result_internals = [node for node in result.traverse() if node.children]

    for node in result_internals:
        if random.random() < recombination_rate:
            # Find a compatible subtree from tree2 to swap in
            node_taxa = set(node.get_leaf_names())

            # Find nodes in tree2 that have overlapping taxa
            compatible_nodes = []
            for t2_node in internal_nodes2:
                t2_taxa = set(t2_node.get_leaf_names())
                overlap = node_taxa.intersection(t2_taxa)
                if (
                    len(overlap) >= 2
                ):  # Need at least 2 taxa overlap for meaningful recombination
                    compatible_nodes.append((t2_node, len(overlap)))

            if compatible_nodes:
                # Choose the node with highest overlap
                compatible_nodes.sort(key=lambda x: x[1], reverse=True)

                # Replace the subtree structure (keep same taxa but change topology)
                # This is a simplified recombination - in reality it would be more complex
                if random.random() < 0.5 and len(node.children) == 2:
                    node.swap_children()


def generate_tree_sequence_with_recombination(
    num_taxa: int = 300, num_trees: int = 10, recombination_rate: float = 0.15
) -> List[Node]:
    """
    Generate a sequence of trees with recombination events.

    Args:
        num_taxa: Number of leaf taxa per tree
        num_trees: Number of trees to generate
        recombination_rate: Rate of recombination between adjacent trees

    Returns:
        List of trees representing a sequence with recombination
    """
    print(f"Generating {num_trees} trees with {num_taxa} taxa each...")

    # Generate taxa names (same for all trees in sequence)
    taxa = generate_taxa_names(num_taxa)

    trees = []

    # Generate first tree randomly
    print("Generating initial tree...")
    first_tree = create_random_binary_tree(taxa)
    trees.append(first_tree)

    # Generate subsequent trees with recombination
    for i in range(1, num_trees):
        print(f"Generating tree {i + 1}/{num_trees} with recombination...")

        # Create a new random tree
        new_tree = create_random_binary_tree(taxa)

        # Simulate recombination with previous tree
        recombined_tree = simulate_recombination_event(
            trees[-1], new_tree, recombination_rate
        )

        trees.append(recombined_tree)

    return trees


def add_realistic_branch_lengths(tree: Node, base_time: float = 1.0) -> None:
    """
    Add realistic branch lengths to a tree using a molecular clock model.

    Args:
        tree: Tree to add branch lengths to
        base_time: Base time for the root
    """

    def set_times_recursive(node: Node, current_time: float) -> None:
        if node.is_leaf():
            # Leaves are at time 0
            node.branch_length = current_time
        else:
            # Internal nodes get times based on coalescent model
            if node.children:
                # Time between coalescent events
                num_lineages = len(node.get_leaf_names())
                if num_lineages > 1:
                    # Coalescent waiting time is exponentially distributed
                    waiting_time = random.expovariate(
                        num_lineages * (num_lineages - 1) / 2
                    )
                    child_time = current_time - waiting_time

                    for child in node.children:
                        child.branch_length = waiting_time
                        set_times_recursive(child, child_time)

    set_times_recursive(tree, base_time)


def save_trees_to_files(
    trees: List[Node], output_dir: str = "large_trees_output"
) -> None:
    """
    Save trees to individual Newick files.

    Args:
        trees: List of trees to save
        output_dir: Directory to save files in
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"Saving {len(trees)} trees to {output_dir}/")

    for i, tree in enumerate(trees):
        filename = output_path / f"tree_{i:03d}.newick"
        try:
            newick_str = write_newick(tree)
            with open(filename, "w") as f:
                f.write(newick_str + "\n")
            print(f"Saved {filename}")
        except Exception as e:
            print(f"Error saving tree {i}: {e}")
            # Fallback: save a simple representation
            with open(filename, "w") as f:
                f.write(f"({','.join(tree.get_leaf_names())});\n")


def generate_statistics(trees: List[Node]) -> None:
    """Print statistics about the generated trees."""
    if not trees:
        return

    print("\n" + "=" * 50)
    print("TREE SEQUENCE STATISTICS")
    print("=" * 50)

    print(f"Number of trees: {len(trees)}")
    print(f"Number of taxa per tree: {len(trees[0].get_leaf_names())}")

    # Calculate tree statistics
    num_internal_nodes = []
    tree_depths = []

    for tree in trees:
        internals = len([node for node in tree.traverse() if node.children])
        num_internal_nodes.append(internals)
        tree_depths.append(tree.get_depth())

    print(
        f"Internal nodes per tree: {min(num_internal_nodes)}-{max(num_internal_nodes)} (avg: {sum(num_internal_nodes) / len(num_internal_nodes):.1f})"
    )
    print(
        f"Tree depth range: {min(tree_depths)}-{max(tree_depths)} (avg: {sum(tree_depths) / len(tree_depths):.1f})"
    )

    # Check if all trees have same taxa
    first_taxa = set(trees[0].get_leaf_names())
    same_taxa = all(set(tree.get_leaf_names()) == first_taxa for tree in trees)
    print(f"All trees have same taxa: {same_taxa}")


def main():
    """Main function to generate large trees with recombination."""
    print("BranchArchitect Large Tree Generator")
    print("=" * 40)

    # Parameters
    num_taxa = 300
    num_trees = 10
    recombination_rate = 0.15

    # Set random seed for reproducibility
    random.seed(42)

    # Generate tree sequence
    trees = generate_tree_sequence_with_recombination(
        num_taxa=num_taxa, num_trees=num_trees, recombination_rate=recombination_rate
    )

    # Add realistic branch lengths
    print("Adding realistic branch lengths...")
    for tree in trees:
        add_realistic_branch_lengths(tree)

    # Generate and print statistics
    generate_statistics(trees)

    # Save trees to files
    save_trees_to_files(trees)

    print(f"\nCompleted! Generated {len(trees)} trees with {num_taxa} taxa each.")
    print("Trees saved to 'large_trees_output/' directory.")

    # Save a summary file
    with open("large_trees_output/README.txt", "w") as f:
        f.write("Large Tree Dataset\n")
        f.write("==================\n\n")
        f.write(f"Generated on: {str(Path().cwd())}\n")
        f.write(f"Number of trees: {num_trees}\n")
        f.write(f"Taxa per tree: {num_taxa}\n")
        f.write(f"Recombination rate: {recombination_rate}\n")
        f.write("Random seed: 42\n\n")
        f.write("Files:\n")
        for i in range(num_trees):
            f.write(f"  tree_{i:03d}.newick\n")


if __name__ == "__main__":
    main()
