"""
Diagnostic to check the reordering logic in detail
"""

from brancharchitect.io import read_newick
from brancharchitect.jumping_taxa.lattice.compute_pivot_solutions_with_deletions import (
    compute_pivot_solutions_with_deletions,
)
from brancharchitect.tree_interpolation.pair_interpolation import discover_pivot_split


def diagnose_reordering_logic():
    """Check if the reordering logic produces correct results"""

    print("=" * 80)
    print("Reordering Logic Diagnostic")
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
    jumping_subtree_solutions, deleted_taxa = compute_pivot_solutions_with_deletions(tree1, tree2)

    # Get ordered s-edges (now using correct leaves-to-root ordering)
    active_split_data = discover_pivot_split(tree1, tree2)
    ordered_edges = active_split_data.get_sorted_edges(
        use_reference=False,
        ascending=True,  # Leaves to root (FIXED!)
    )

    print(f"\nProcessing Order (Leaves to Root):")
    for i, s_edge in enumerate(ordered_edges, 1):
        indices = s_edge.resolve_to_indices()
        taxa = sorted([taxa_reverse[idx] for idx in indices])
        print(
            f"  {i}. Size={len(indices):2d}, Taxa={{{', '.join(taxa[:5])}{', ...' if len(taxa) > 5 else ''}}}"
        )

    # For each s-edge, check the reordering logic
    print("\n" + "=" * 80)
    print("Checking Reordering Logic for Each S-Edge")
    print("=" * 80)

    for i, s_edge in enumerate(ordered_edges, 1):
        indices = s_edge.resolve_to_indices()
        taxa = sorted([taxa_reverse[idx] for idx in indices])

        print(f"\n{'=' * 80}")
        print(
            f"S-Edge {i}: {{{', '.join(taxa[:5])}{', ...' if len(taxa) > 5 else ''}}}"
        )
        print(f"{'=' * 80}")

        # Get the source and destination subtrees
        source_subtree = tree1.find_node_by_split(s_edge)
        dest_subtree = tree2.find_node_by_split(s_edge)

        if source_subtree is None or dest_subtree is None:
            print("  ⚠️  S-edge not found in one of the trees!")
            continue

        source_order = list(source_subtree.get_current_order())
        destination_order = list(dest_subtree.get_current_order())

        print(f"\nSource order ({len(source_order)} taxa):")
        print(f"  {source_order}")

        print(f"\nDestination order ({len(destination_order)} taxa):")
        print(f"  {destination_order}")

        # Get the jumping taxa for this s-edge
    partitions = jumping_subtree_solutions.get(s_edge, [])
        if not partitions:
            print("\n  ℹ️  No jumping taxa for this s-edge (no reordering needed)")
            continue

        print(f"\nJumping taxa partitions ({len(partitions)}):")
        for j, partition in enumerate(partitions, 1):
            part_indices = partition.resolve_to_indices()
            part_taxa = sorted([taxa_reverse[idx] for idx in part_indices])
            print(f"  Partition {j}: {{{', '.join(part_taxa)}}}")

                # Simulate the reordering logic
                print(
                    f"\n  Analyzing reordering for moving subtree: {{{', '.join(part_taxa)}}}"
                )

                mover_leaves = set(part_taxa)

                # Check if mover leaves are in source order
                if not mover_leaves.issubset(set(source_order)):
                    print(f"    ❌ ERROR: Mover leaves not in source order!")
                    continue

                # Step 1: Isolate anchors
                source_anchors = [
                    taxon for taxon in source_order if taxon not in mover_leaves
                ]
                print(f"\n    1. Anchors (non-moving taxa): {source_anchors}")

                # Step 2: Find insertion point
                num_anchors_before = 0
                insertion_found = False
                for taxon in destination_order:
                    if taxon in mover_leaves:
                        insertion_found = True
                        break
                    if taxon in source_anchors:
                        num_anchors_before += 1

                print(f"\n    2. Insertion point calculation:")
                print(f"       - Scanning destination order: {destination_order}")
                print(
                    f"       - Number of anchors before mover block: {num_anchors_before}"
                )

                if insertion_found:
                    print(
                        f"       - Mover block starts at position {num_anchors_before} (after {num_anchors_before} anchors)"
                    )
                else:
                    print(
                        f"       ⚠️  WARNING: Mover block not found in destination order!"
                    )

                # Step 3: Construct new order
                mover_block_ordered = [
                    taxon for taxon in source_order if taxon in mover_leaves
                ]
                new_order = source_anchors[:]
                new_order[num_anchors_before:num_anchors_before] = mover_block_ordered

                print(f"\n    3. Constructing new order:")
                print(f"       - Mover block (source order): {mover_block_ordered}")
                print(f"       - New order: {new_order}")

                # Check if new order matches destination
                if new_order == destination_order:
                    print(f"\n    ✅ SUCCESS: New order matches destination!")
                else:
                    print(f"\n    ❌ PROBLEM: New order does NOT match destination!")
                    print(f"       Expected: {destination_order}")
                    print(f"       Got:      {new_order}")

                    # Analyze the difference
                    print(f"\n    🔍 Analyzing difference:")
                    for k, (expected, got) in enumerate(
                        zip(destination_order, new_order)
                    ):
                        if expected != got:
                            print(
                                f"       Position {k}: Expected '{expected}', got '{got}' ❌"
                            )
                        else:
                            print(f"       Position {k}: '{expected}' ✓")

                    # Check if the issue is internal ordering of mover block
                    dest_mover_order = [
                        t for t in destination_order if t in mover_leaves
                    ]
                    if dest_mover_order != mover_block_ordered:
                        print(f"\n    ⚠️  INTERNAL ORDER MISMATCH:")
                        print(f"       Mover block in destination: {dest_mover_order}")
                        print(
                            f"       Mover block in source:      {mover_block_ordered}"
                        )
                        print(
                            f"\n    💡 The problem: We're preserving source order for mover block,"
                        )
                        print(f"       but destination has a DIFFERENT internal order!")


if __name__ == "__main__":
    diagnose_reordering_logic()
