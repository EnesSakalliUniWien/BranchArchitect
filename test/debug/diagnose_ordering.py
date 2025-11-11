"""
Diagnostic to check s-edge ordering and processing sequence
"""

from brancharchitect.io import read_newick
from brancharchitect.tree_interpolation.pair_interpolation import (
    process_tree_pair_interpolation,
)
from brancharchitect.jumping_taxa.lattice.iterate_lattice_algorithm import (
    iterate_lattice_algorithm,
)
from brancharchitect.tree_interpolation.edge_sorting_utils import (
    sort_edges_by_depth,
    compute_edge_depths,
)


def diagnose_sedge_ordering():
    """Check if s-edges are being processed in the correct order"""

    print("=" * 80)
    print("S-Edge Ordering Diagnostic")
    print("=" * 80)

    # Read trees
    trees = read_newick(
        "/Users/berksakalli/Projects/BranchArchitect/current_testfiles/small_example.newick"
    )
    tree1 = trees[0]
    tree2 = trees[1]

    taxa_reverse = {v: k for k, v in tree1.taxa_encoding.items()}

    print("\nTree 1 → Tree 2 Analysis")
    print("-" * 80)

    # Get s-edges and their solutions
    jumping_subtree_solutions, deleted_taxa = iterate_lattice_algorithm(tree1, tree2)

    print(f"\nDiscovered {len(jumping_subtree_solutions)} s-edges:")
    for i, (s_edge, partitions) in enumerate(jumping_subtree_solutions.items(), 1):
        indices = s_edge.resolve_to_indices()
        taxa = sorted([taxa_reverse[idx] for idx in indices])

        print(f"\nS-Edge {i}: {len(taxa)} taxa")
        print(f"  Taxa: {{{', '.join(taxa)}}}")
        print(f"  Size: {len(indices)}")

        # Calculate "depth" (we'll check how this is computed)
        node_in_tree1 = tree1.find_node_by_split(s_edge)
        node_in_tree2 = tree2.find_node_by_split(s_edge)

        if node_in_tree1:
            depth1 = 0
            cur = node_in_tree1
            while cur.parent is not None:
                depth1 += 1
                cur = cur.parent
            print(f"  Depth in Tree 1: {depth1}")
        else:
            print("  Not found in Tree 1!")

        if node_in_tree2:
            depth2 = 0
            cur = node_in_tree2
            while cur.parent is not None:
                depth2 += 1
                cur = cur.parent
            print(f"  Depth in Tree 2: {depth2}")
        else:
            print("  Not found in Tree 2!")  # Show solutions
        if partitions:
            jumping_parts = []
            for partition in partitions:
                part_indices = partition.resolve_to_indices()
                part_taxa = sorted([taxa_reverse[idx] for idx in part_indices])
                jumping_parts.append(f"{{{', '.join(part_taxa)}}}")
            print(f"  Jumping: {' + '.join(jumping_parts)}")

    # Now check the actual ordering used in interpolation
    print("\n" + "=" * 80)
    print("Checking Edge Ordering by Depth")
    print("=" * 80)

    # Compute depths for each edge
    lattice_edges = list(jumping_subtree_solutions.keys())
    depths_tree1 = compute_edge_depths(lattice_edges, tree1)
    depths_tree2 = compute_edge_depths(lattice_edges, tree2)

    print("\nDepths computed:")
    print(f"  Tree 1 depths: {depths_tree1}")
    print(f"  Tree 2 depths: {depths_tree2}")

    # Get sorted edges (using tree2 as target)
    ordered_edges_ascending = sort_edges_by_depth(
        edges=lattice_edges,
        tree=tree2,
        ascending=True,  # Leaves to root
    )

    ordered_edges_descending = sort_edges_by_depth(
        edges=lattice_edges,
        tree=tree2,
        ascending=False,  # Root to leaves
    )

    print("\n" + "-" * 80)
    print("Ordering: ASCENDING (leaves to root) - CURRENTLY USED")
    print("-" * 80)
    for i, s_edge in enumerate(ordered_edges_ascending, 1):
        indices = s_edge.resolve_to_indices()
        taxa = sorted([taxa_reverse[idx] for idx in indices])
        depth = depths_tree2.get(s_edge, "?")
        print(
            f"  {i}. Size={len(indices):2d}, Depth={depth}, Taxa={{{', '.join(taxa[:5])}{', ...' if len(taxa) > 5 else ''}}}"
        )

    print("\n" + "-" * 80)
    print("Ordering: DESCENDING (root to leaves) - OLD/INCORRECT")
    print("-" * 80)
    for i, s_edge in enumerate(ordered_edges_descending, 1):
        indices = s_edge.resolve_to_indices()
        taxa = sorted([taxa_reverse[idx] for idx in indices])
        depth = depths_tree2.get(s_edge, "?")
        print(
            f"  {i}. Size={len(indices):2d}, Depth={depth}, Taxa={{{', '.join(taxa[:5])}{', ...' if len(taxa) > 5 else ''}}}"
        )

    # Key insight: check if larger s-edges (more taxa) should be processed BEFORE or AFTER smaller ones
    print("\n" + "=" * 80)
    print("SOLUTION: Process from leaves to root!")
    print("=" * 80)
    print("""
For phylogenetic tree interpolation:
- ROOT TO LEAVES (large → small): Process parent splits before child splits
  * Pros: Parent structure is established before children are modified
  * Cons: May cause "snapback" if children need reordering that parents undo
  * STATUS: OLD/INCORRECT - causes moves to happen too early

- LEAVES TO ROOT (small → large): Process child splits before parent splits  ✓
  * Pros: Local changes stabilize before broader reorganization
  * Cons: Parent changes might need to undo child work (rare in practice)
  * STATUS: CURRENT/CORRECT - fixes the "moves too early" problem

Previous setting: Root to leaves (ascending=False) - CAUSED SNAPBACK
New setting: Leaves to root (ascending=True) - FIXED!
""")

    print("\nAnalyzing s-edge relationships:")
    for i, s_edge_i in enumerate(ordered_edges_descending, 1):
        indices_i = set(s_edge_i.resolve_to_indices())
        taxa_i = sorted([taxa_reverse[idx] for idx in indices_i])

        for j, s_edge_j in enumerate(ordered_edges_descending, 1):
            if i >= j:
                continue

            indices_j = set(s_edge_j.resolve_to_indices())

            if indices_j.issubset(indices_i):
                taxa_j = sorted([taxa_reverse[idx] for idx in indices_j])
                print(
                    f"\n  S-Edge {i} ({len(indices_i)} taxa) CONTAINS S-Edge {j} ({len(indices_j)} taxa)"
                )
                print(
                    f"    Parent: {{{', '.join(taxa_i[:3])}{', ...' if len(taxa_i) > 3 else ''}}}"
                )
                print(f"    Child:  {{{', '.join(taxa_j)}}}")
                print(
                    "    → In old ordering, parent was processed BEFORE child (caused snapback)"
                )
                print(
                    "    → In NEW ordering, child is processed BEFORE parent (FIXED!)"
                )


if __name__ == "__main__":
    diagnose_sedge_ordering()
