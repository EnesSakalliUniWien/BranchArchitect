import json
import sys
import os
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.jumping_taxa.verification import verify_jumping_taxa_solution

# Setup paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def check_file(path):
    print(f"Checking {path}")
    with open(path) as f:
        data = json.load(f)

    t1 = parse_newick(data["tree1"]);
    if isinstance(t1, list): t1 = t1[0]
    t2 = parse_newick(data["tree2"], encoding=t1.taxa_encoding)
    if isinstance(t2, list): t2 = t2[0]

    for i, sol in enumerate(data['solutions']):
        flat = []
        for p in sol: flat.extend(p)
        res = verify_jumping_taxa_solution(t1, t2, flat)
        print(f"  Sol {i+1} {flat}: {'PASS' if res['success'] else 'FAIL'}")

check_file("test/colouring/trees/heiko_test_130825/heiko_test_130825.json")
check_file("test/colouring/trees/simon_test_tree_3/simon_test_tree_3.json")
check_file("test/colouring/trees/simon_test_tree_3/reverse_simon_test_tree_3.json")
