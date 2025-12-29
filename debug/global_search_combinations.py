"""
Exhaustively search for ANY set of taxa (subset of ALL leaves) that makes trees isomorphic.
Candidates: All taxa except O1, O2 (anchors).
Base removals: A2, B1 (We keep these fixed as they are obvious jumpers, to reduce search space complexity from 2^15 to 2^13).
"""

from itertools import combinations
from brancharchitect.parser.newick_parser import parse_newick as parse

tree1_newick = "((O1,O2),(((((A,A1),A2),(B,B1)),C),((D,(E,(((F,G),I),M))),H)));"
tree2_newick = "((O1,O2),(((((A,A1),B1),(B,A2)),(C,(E,(((F,M),I),G)))),(D,H)));"

base_removals = ["A2", "B1"]

# Get all leaves from tree 1
temp_t1 = parse(tree1_newick)
all_leaves = [l.name for l in temp_t1.get_leaves()]

# Candidates: All leaves except anchors and base removals
candidates = [n for n in all_leaves if n not in ["O1", "O2"] and n not in base_removals]


def check_solution(subset):
    t1 = parse(tree1_newick)
    t2 = parse(tree2_newick)

    # UNIFY ENCODING
    t2.initialize_split_indices(t1.taxa_encoding)

    # Full removal set
    to_remove = base_removals + list(subset)

    encoding = t1.taxa_encoding
    indices = [encoding[n] for n in to_remove if n in encoding]

    t1.delete_taxa(indices)
    t2.delete_taxa(indices)

    return t1 == t2


print(f"Global Search Candidates ({len(candidates)}): sorted(candidates)")
print(candidates)
print(f"Base removals (always included): {base_removals}")
print("-" * 60)

# We are looking for a solution smaller than the known best (size 5).
# Algorithm solution: {E, F, G, I, M} (Size 5)
found_better = False

for r in range(1, 6):
    print(f"Checking all subsets of size {r}...")
    count = 0
    for subset in combinations(candidates, r):
        count += 1
        if check_solution(subset):
            print(f"!!! FOUND SOLUTION: {subset} (Size {r})")
            if r < 5:
                print("    ^ BETTER than algorithm solution!")
                found_better = True
            else:
                print("    ^ Equal size to algorithm solution.")
    print(f"  Checked {count} combinations.")

if not found_better:
    print("\nCONCLUSION: No solution found with fewer than 5 additional jumping taxa.")
else:
    print("\nCONCLUSION: Found a more parsimonious solution!")
