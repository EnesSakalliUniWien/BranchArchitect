from brancharchitect.io import read_newick
from brancharchitect.rooting.rooting import simple_reroot, reroot_at_node
import os


def test_rerooting_with_six_taxa():
    # Load the trees from the six_taxa_all_permutations.newick file
    data_path = os.path.join(
        os.path.dirname(__file__), "..", "six_taxa_all_permutations.newick"
    )
    trees = read_newick(data_path)
    # Use the first two trees for rerooting tests
    tree1 = trees[0]
    tree2 = trees[1]

    # Pick a non-root node from tree1 (e.g., the first internal node)
    internal_nodes = [n for n in tree1.traverse() if not n.is_leaf()]
    assert internal_nodes, "No internal nodes found in tree1!"
    ref_node = internal_nodes[0]

    # Reroot tree2 to best match the reference node from tree1
    rerooted_tree2 = simple_reroot(tree2, ref_node)
    assert rerooted_tree2 is not None
    # Check that rerooted_tree2 is a Node and has the same taxa as tree2
    taxa1 = sorted([leaf.name for leaf in tree1.leaves])
    taxa2 = sorted([leaf.name for leaf in rerooted_tree2.leaves])
    assert taxa1 == taxa2

    # Reroot at a leaf for sanity
    leaf = [n for n in tree1.traverse() if n.is_leaf()][0]
    rerooted_leaf = reroot_at_node(leaf)
    assert rerooted_leaf is leaf

    # Try rerooting tree2 at the same leaf name
    leaf2 = [n for n in tree2.traverse() if n.name == leaf.name][0]
    rerooted_leaf2 = reroot_at_node(leaf2)
    assert rerooted_leaf2 is leaf2
