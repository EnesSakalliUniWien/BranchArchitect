#!/usr/bin/env python3
"""
Script to generate large trees with high number of leaf labels (300) and recombination events using tskit.
Creates tree sequences compatible with population genetics simulations and exports to formats
compatible with BranchArchitect.
"""

import tskit
import msprime
import numpy as np
from typing import Optional
from pathlib import Path
import json


def generate_tree_sequence_with_recombination(
    num_samples: int = 300,
    sequence_length: float = 1e6,  # 1 Mb sequence
    population_size: int = 10000,
    recombination_rate: float = 1e-8,  # Per base pair per generation
    mutation_rate: float = 1e-8,  # Per base pair per generation
    random_seed: Optional[int] = None,
) -> tskit.TreeSequence:
    """
    Generate a tree sequence with recombination using msprime simulator.

    Args:
        num_samples: Number of samples (leaf nodes) per tree
        sequence_length: Length of the genomic sequence
        population_size: Effective population size
        recombination_rate: Recombination rate per base pair per generation
        mutation_rate: Mutation rate per base pair per generation
        random_seed: Random seed for reproducibility

    Returns:
        TreeSequence object containing trees with recombination
    """
    print(f"Generating tree sequence with {num_samples} samples...")
    print(f"Sequence length: {sequence_length:,.0f} bp")
    print(f"Population size: {population_size:,}")
    print(f"Recombination rate: {recombination_rate}")
    print(f"Mutation rate: {mutation_rate}")

    # Generate ancestry (coalescent simulation with recombination)
    ts = msprime.sim_ancestry(
        samples=num_samples,
        population_size=population_size,
        sequence_length=sequence_length,
        recombination_rate=recombination_rate,
        random_seed=random_seed,
    )

    # Add mutations
    ts = msprime.sim_mutations(ts, rate=mutation_rate, random_seed=random_seed)

    print(f"Generated tree sequence with {ts.num_trees} trees")
    print(f"Total mutations: {ts.num_mutations}")
    print(f"Sequence length: {ts.sequence_length:,.0f}")

    return ts


def generate_demographic_model_tree_sequence(
    num_samples: int = 300,
    sequence_length: float = 1e6,
    random_seed: Optional[int] = None,
) -> tskit.TreeSequence:
    """
    Generate a tree sequence with a more complex demographic model.
    Uses a simplified approach for compatibility.

    Args:
        num_samples: Number of samples per tree
        sequence_length: Length of genomic sequence
        random_seed: Random seed for reproducibility

    Returns:
        TreeSequence with different population parameters
    """
    print("Generating tree sequence with demographic model...")

    # Use a simpler approach: just different population sizes
    # This creates a bottleneck effect with more coalescence
    ts = msprime.sim_ancestry(
        samples=num_samples,
        population_size=1000,  # Smaller population size creates more coalescence
        sequence_length=sequence_length,
        recombination_rate=2e-8,  # Higher recombination rate
        random_seed=random_seed,
    )

    # Add mutations
    ts = msprime.sim_mutations(ts, rate=2e-8, random_seed=random_seed)

    print(f"Generated demographic model with {ts.num_trees} trees")
    return ts


def analyze_tree_sequence_statistics(ts: tskit.TreeSequence) -> dict:
    """
    Analyze and return statistics about the tree sequence.

    Args:
        ts: TreeSequence to analyze

    Returns:
        Dictionary containing various statistics
    """
    stats = {
        "num_trees": ts.num_trees,
        "num_samples": ts.num_samples,
        "num_nodes": ts.num_nodes,
        "num_mutations": ts.num_mutations,
        "sequence_length": ts.sequence_length,
        "tree_heights": [],
        "num_internal_nodes_per_tree": [],
        "tree_spans": [],
        "recombination_breakpoints": [],
    }

    # Analyze each tree
    for tree in ts.trees():
        # Tree height (time to root)
        stats["tree_heights"].append(tree.time(tree.root))

        # Number of internal nodes
        internal_nodes = sum(1 for node in tree.nodes() if not tree.is_sample(node))
        stats["num_internal_nodes_per_tree"].append(internal_nodes)

        # Tree span (genomic region)
        stats["tree_spans"].append(tree.span)

    # Find recombination breakpoints
    breakpoints = list(ts.breakpoints())
    stats["recombination_breakpoints"] = breakpoints[1:-1]  # Exclude start and end

    # Summary statistics
    stats["avg_tree_height"] = np.mean(stats["tree_heights"])
    stats["max_tree_height"] = np.max(stats["tree_heights"])
    stats["min_tree_height"] = np.min(stats["tree_heights"])
    stats["avg_internal_nodes"] = np.mean(stats["num_internal_nodes_per_tree"])
    stats["num_recombination_events"] = len(stats["recombination_breakpoints"])

    return stats


def export_trees_to_newick(
    ts: tskit.TreeSequence, output_dir: str = "tskit_trees_output", max_trees: int = 50
) -> None:
    """
    Export trees from tree sequence to Newick format files.

    Args:
        ts: TreeSequence to export
        output_dir: Directory to save files
        max_trees: Maximum number of trees to export (to avoid too many files)
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print(f"Exporting trees to {output_dir}/")

    # Sample trees evenly across the sequence if there are too many
    tree_indices = np.linspace(
        0, ts.num_trees - 1, min(max_trees, ts.num_trees), dtype=int
    )

    exported_count = 0
    for i, tree in enumerate(ts.trees()):
        if i not in tree_indices:
            continue

        # Create sample labels (T001, T002, etc.)
        sample_labels = {sample: f"T{sample + 1:03d}" for sample in ts.samples()}

        # Export to Newick format
        newick_str = tree.newick(node_labels=sample_labels)

        # Save to file
        filename = (
            output_path
            / f"tree_{exported_count:03d}_pos_{tree.interval.left:.0f}-{tree.interval.right:.0f}.newick"
        )
        with open(filename, "w") as f:
            f.write(newick_str + "\n")

        exported_count += 1
        if exported_count % 10 == 0:
            print(f"Exported {exported_count}/{len(tree_indices)} trees...")

    print(f"Exported {exported_count} trees total")


def save_tree_sequence_metadata(
    ts: tskit.TreeSequence, stats: dict, output_dir: str = "tskit_trees_output"
) -> None:
    """
    Save metadata and statistics about the tree sequence.

    Args:
        ts: TreeSequence object
        stats: Statistics dictionary
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save tree sequence in native format
    ts.dump(output_path / "tree_sequence.trees")
    print(f"Saved tree sequence to {output_path / 'tree_sequence.trees'}")

    # Save statistics as JSON
    # Convert numpy types to Python types for JSON serialization
    json_stats = {}
    for key, value in stats.items():
        if isinstance(value, np.ndarray):
            json_stats[key] = value.tolist()
        elif isinstance(value, np.floating):
            json_stats[key] = float(value)
        elif isinstance(value, np.integer):
            json_stats[key] = int(value)
        else:
            json_stats[key] = value

    with open(output_path / "statistics.json", "w") as f:
        json.dump(json_stats, f, indent=2)

    # Save human-readable summary
    with open(output_path / "README.txt", "w") as f:
        f.write("Tree Sequence Dataset (Generated with tskit)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of trees: {stats['num_trees']}\n")
        f.write(f"Number of samples per tree: {stats['num_samples']}\n")
        f.write(f"Total nodes: {stats['num_nodes']}\n")
        f.write(f"Total mutations: {stats['num_mutations']}\n")
        f.write(f"Sequence length: {stats['sequence_length']:,.0f} bp\n")
        f.write(
            f"Number of recombination events: {stats['num_recombination_events']}\n\n"
        )

        f.write("Tree Statistics:\n")
        f.write(f"  Average tree height: {stats['avg_tree_height']:.2f} generations\n")
        f.write(
            f"  Tree height range: {stats['min_tree_height']:.2f} - {stats['max_tree_height']:.2f}\n"
        )
        f.write(
            f"  Average internal nodes per tree: {stats['avg_internal_nodes']:.1f}\n\n"
        )

        f.write("Files:\n")
        f.write("  tree_sequence.trees - Native tskit format\n")
        f.write("  statistics.json - Detailed statistics\n")
        f.write("  tree_*.newick - Individual trees in Newick format\n")


def demonstrate_tree_sequence_features(ts: tskit.TreeSequence) -> None:
    """
    Demonstrate some key features of the tree sequence.

    Args:
        ts: TreeSequence to demonstrate
    """
    print("\n" + "=" * 60)
    print("TREE SEQUENCE FEATURES DEMONSTRATION")
    print("=" * 60)

    # Show first few trees
    print("\nFirst 3 trees in the sequence:")
    for i, tree in enumerate(ts.trees()):
        if i >= 3:
            break
        print(
            f"\nTree {i + 1} (position {tree.interval.left:.0f}-{tree.interval.right:.0f}):"
        )
        print(f"  Root: {tree.root}, Height: {tree.time(tree.root):.2f} generations")
        print(f"  Samples under root: {len(list(tree.samples(tree.root)))}")

        # Show MRCA of first few samples
        if ts.num_samples >= 3:
            mrca = tree.mrca(0, 1, 2)
            print(f"  MRCA of samples 0,1,2: node {mrca}, time {tree.time(mrca):.2f}")

    # Show some mutations
    if ts.num_mutations > 0:
        print(f"\nFirst 5 mutations (out of {ts.num_mutations} total):")
        for i, mutation in enumerate(ts.mutations()):
            if i >= 5:
                break
            # Get site position from the site table
            site = ts.site(mutation.site)
            print(
                f"  Mutation {i + 1}: position {site.position:.0f}, node {mutation.node}"
            )

    # Show recombination breakpoints
    breakpoints = list(ts.breakpoints())
    if len(breakpoints) > 2:
        print(f"\nRecombination breakpoints: {len(breakpoints) - 2} events")
        print(f"  First few breakpoints: {[int(x) for x in breakpoints[1:6]]}")


def main():
    """Main function to generate large trees with tskit."""
    print("BranchArchitect tskit Tree Generator")
    print("=" * 40)

    # Parameters
    num_samples = 300
    sequence_length = 1e6  # 1 Mb
    random_seed = 42

    # Generate basic tree sequence with recombination
    print("\n1. Generating basic tree sequence...")
    ts_basic = generate_tree_sequence_with_recombination(
        num_samples=num_samples,
        sequence_length=sequence_length,
        recombination_rate=1e-8,
        random_seed=random_seed,
    )

    # Generate tree sequence with demographic model
    print("\n2. Generating tree sequence with demographic model...")
    ts_demo = generate_demographic_model_tree_sequence(
        num_samples=num_samples,
        sequence_length=sequence_length,
        random_seed=random_seed,
    )

    # Analyze statistics for both
    print("\n3. Analyzing statistics...")
    stats_basic = analyze_tree_sequence_statistics(ts_basic)
    stats_demo = analyze_tree_sequence_statistics(ts_demo)

    print(
        f"\nBasic model: {stats_basic['num_trees']} trees, {stats_basic['num_recombination_events']} recombinations"
    )
    print(
        f"Demographic model: {stats_demo['num_trees']} trees, {stats_demo['num_recombination_events']} recombinations"
    )

    # Save basic model results
    print("\n4. Exporting basic model...")
    export_trees_to_newick(ts_basic, "tskit_basic_output", max_trees=20)
    save_tree_sequence_metadata(ts_basic, stats_basic, "tskit_basic_output")

    # Save demographic model results
    print("\n5. Exporting demographic model...")
    export_trees_to_newick(ts_demo, "tskit_demographic_output", max_trees=20)
    save_tree_sequence_metadata(ts_demo, stats_demo, "tskit_demographic_output")

    # Demonstrate features
    demonstrate_tree_sequence_features(ts_basic)

    print(f"\n✅ Completed! Generated tree sequences with {num_samples} samples each.")
    print("Results saved to:")
    print("  - tskit_basic_output/ (basic recombination model)")
    print("  - tskit_demographic_output/ (demographic model with bottleneck)")
    print("\nTo load a tree sequence later:")
    print("  import tskit")
    print("  ts = tskit.load('tskit_basic_output/tree_sequence.trees')")


if __name__ == "__main__":
    main()
