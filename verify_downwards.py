from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.tree import Node

t1_str = "((((A1,(A2,A3)),(X1,X2)),(B1,B2)),(O1,O2));"
t2_str = "(((A1,(A2,(A3,(X1,X2)))),(B1,B2)),(O1,O2));"

t1 = parse_newick(t1_str)
t2 = parse_newick(t2_str)

# Unified encoding
taxa_t1 = {l.name for l in t1.get_leaves()}
all_taxa = sorted(list(taxa_t1))
shared_encoding = {name: i for i, name in enumerate(all_taxa)}
t1.initialize_split_indices(shared_encoding)
t2.initialize_split_indices(shared_encoding)

def check_solution(deleted_names, t1, t2):
    c1 = t1.deep_copy()
    c2 = t2.deep_copy()

    indices = [shared_encoding[n] for n in deleted_names]
    c1.delete_taxa(indices)
    c2.delete_taxa(indices)

    is_equal = (c1 == c2)
    print(f"Deleting {deleted_names}: Trees Equal? {is_equal}")
    if not is_equal:
        print("  T1:", c1.to_newick(lengths=False))
        print("  T2:", c2.to_newick(lengths=False))

print("=== Verifying Expected Solutions ===")
check_solution(["A2"], t1, t2)
check_solution(["A3"], t1, t2)
check_solution(["X1", "X2"], t1, t2)

print("\n=== Verifying My Solution ===")
check_solution(["A1", "A2"], t1, t2)
check_solution(["A1", "A3"], t1, t2)
