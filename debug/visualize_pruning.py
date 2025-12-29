"""
Visualize the tree topology (Newick strings) after applying different removal strategies.
This proves whether the removal logic and solution logic are correct.
"""

from brancharchitect.parser.newick_parser import parse_newick as parse

tree1_newick = "((O1,O2),(((((A,A1),A2),(B,B1)),C),((D,(E,(((F,G),I),M))),H)));"
tree2_newick = "((O1,O2),(((((A,A1),B1),(B,A2)),(C,(E,(((F,M),I),G)))),(D,H)));"


def visualize_strategy(name, removals):
    print(f"\n{'=' * 20} STRATEGY: {name} {'=' * 20}")
    print(f"Removing: {removals}")

    t1 = parse(tree1_newick)
    t2 = parse(tree2_newick)

    # Sync encoding
    t2.initialize_split_indices(t1.taxa_encoding)

    # Delete
    encoding = t1.taxa_encoding
    indices = [encoding[n] for n in removals if n in encoding]
    t1.delete_taxa(indices)
    t2.delete_taxa(indices)

    # Print simplified Newick (structure only, no lengths/metadata)
    print(f"\nTree 1 Result: {t1.to_newick(lengths=False)}")
    print(f"Tree 2 Result: {t2.to_newick(lengths=False)}")

    if t1 == t2:
        print("\nSUCCESS: Trees are IDENTICAL.")
    else:
        print("\nFAILURE: Trees are DIFFERENT.")


# 1. Algorithm Solution
visualize_strategy("REMOVE BLOCK (E-M)", ["A2", "B1", "E", "F", "G", "I", "M"])

# 2. D Solution
visualize_strategy("REMOVE D + SCATTERED", ["A2", "B1", "D", "C", "F", "G", "H"])
# Note: Removals for D solution (from earlier global search): {A2, B1, C, D, F, G, H}

# 3. D Only (to show the mismatch user asked about)
visualize_strategy("REMOVE D ONLY", ["A2", "B1", "D", "M"])
# (Included M because it's in base removals of previous steps/algo context)
