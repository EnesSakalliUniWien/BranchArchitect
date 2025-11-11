"""
Diagnostic script to inspect s-edges and solutions returned by iterate_lattice_algorithm
"""

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.jumping_taxa.lattice.iterate_lattice_algorithm import (
    iterate_lattice_algorithm,
)
from brancharchitect.jumping_taxa.lattice.pivot_edge_solver import lattice_algorithm


def inspect_sedges_and_solutions():
    """Inspect s-edges and their solutions in detail"""

    # Test case 1: Simple trees
    print("=" * 80)
    print("TEST CASE 1: Simple Trees")
    print("=" * 80)
    tree1_str = "((((A,B),(C,D)),O));"
    tree2_str = "((((A,C),(B,D)),O));"

    trees = parse_newick(tree1_str + tree2_str)
    tree1 = trees[0]
    tree2 = trees[1]

    print(f"\nTree 1: {tree1_str}")
    print(f"Tree 2: {tree2_str}")

    # Call iterate_lattice_algorithm
    print("\n--- Calling iterate_lattice_algorithm ---")
    jumping_subtree_solutions, deleted_taxa_per_iteration = iterate_lattice_algorithm(
        tree1, tree2
    )

    print(f"\nReturned type: {type(jumping_subtree_solutions)}")
    print(f"Number of s-edges: {len(jumping_subtree_solutions)}")
    print(f"Number of iterations with deletions: {len(deleted_taxa_per_iteration)}")

    print("\n--- S-Edge to Solutions Mapping ---")
    # Create reverse mapping
    taxa_reverse = {v: k for k, v in tree1.taxa_encoding.items()}

    for i, (s_edge, partitions) in enumerate(jumping_subtree_solutions.items(), 1):
        print(f"\nS-Edge {i}:")
        print(f"  Partition: {s_edge}")
        print(f"  Bitmask: {s_edge.bitmask}")
        print(f"  Indices: {s_edge.resolve_to_indices()}")
        print(f"  Taxa: {[taxa_reverse[idx] for idx in s_edge.resolve_to_indices()]}")
        print(f"  Number of partitions: {len(partitions)}")

        for k, partition in enumerate(partitions, 1):
            indices = partition.resolve_to_indices()
            taxa = [taxa_reverse[idx] for idx in indices]
            print(
                f"    Partition {k}: {partition} -> indices {indices} -> taxa {taxa}"
            )

    # Also check what lattice_algorithm returns directly
    print("\n\n--- Direct call to lattice_algorithm (single iteration) ---")
    direct_solutions_dict = lattice_algorithm(tree1, tree2)

    print("\nDirect return types:")
    print(f"  solutions type: {type(direct_solutions_dict)}")
    print(f"  Number of pivot edges: {len(direct_solutions_dict)}")

    for i, (split, partitions) in enumerate(direct_solutions_dict.items()):
        print(f"\nPivot {i}:")
        print(f"  Split: {split}")
        print(f"  Split indices: {split.resolve_to_indices()}")
        print(
            f"  Split taxa: {[taxa_reverse[idx] for idx in split.resolve_to_indices()]}"
        )
        print(f"  Partitions type: {type(partitions)}")
        print(f"  Number of partitions: {len(partitions)}")

        for j, partition in enumerate(partitions, 1):
            indices = partition.resolve_to_indices()
            taxa = [taxa_reverse[idx] for idx in indices]
            print(f"    Partition {j}: {partition} -> indices {indices} -> taxa {taxa}")

    # Test case 2: More complex trees
    print("\n\n" + "=" * 80)
    print("TEST CASE 2: Complex Trees")
    print("=" * 80)

    tree1_complex_str = (
        "((O1,O2),(((((A,A1),A2),(B,B1)),C),((D,(E,(((F,G),I),M))),H)));"
    )
    tree2_complex_str = (
        "((O1,O2),(((((A,A1),B1),(B,A2)),(C,(E,(((F,M),I),G)))),(D,H)));"
    )

    trees_complex = parse_newick(tree1_complex_str + tree2_complex_str)
    tree1_complex = trees_complex[0]
    tree2_complex = trees_complex[1]

    print(f"\nTree 1: {tree1_complex_str}")
    print(f"Tree 2: {tree2_complex_str}")

    print("\n--- Calling iterate_lattice_algorithm ---")
    jumping_subtree_solutions_complex, deleted_taxa_complex = iterate_lattice_algorithm(
        tree1_complex, tree2_complex
    )

    print(f"\nNumber of s-edges: {len(jumping_subtree_solutions_complex)}")
    print(f"Number of iterations with deletions: {len(deleted_taxa_complex)}")

    print("\n--- S-Edge Summary ---")
    taxa_reverse_complex = {v: k for k, v in tree1_complex.taxa_encoding.items()}

    for i, (s_edge, partitions) in enumerate(
        jumping_subtree_solutions_complex.items(), 1
    ):
        indices = s_edge.resolve_to_indices()
        taxa = [taxa_reverse_complex[idx] for idx in indices]
        print(f"\nS-Edge {i}: {len(taxa)} taxa")
        print(f"  Taxa: {', '.join(taxa)}")
        print(f"  Partitions: {len(partitions)}")

        jumping_taxa = []
        for partition in partitions:
            part_indices = partition.resolve_to_indices()
            part_taxa = [taxa_reverse_complex[idx] for idx in part_indices]
            jumping_taxa.append(f"{{{', '.join(part_taxa)}}}")
        print(f"    Jumping: {' + '.join(jumping_taxa)}")

    print("\n--- Deleted Taxa Per Iteration ---")
    for i, deleted_set in enumerate(deleted_taxa_complex, 1):
        taxa = [taxa_reverse_complex[idx] for idx in deleted_set]
        print(f"  Iteration {i}: {', '.join(taxa)}")


if __name__ == "__main__":
    inspect_sedges_and_solutions()
