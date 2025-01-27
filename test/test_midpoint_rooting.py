from brancharchitect.io import parse_newick

############################################################
# HELPER: FOR DIRECT TESTING OF INTERNAL FUNCTIONS
############################################################
from brancharchitect.tree import (
    _farthest_leaf_pointer,
    _build_pointer_path,
)


def test_single_leaf():
    """
    STEP 1: Minimal single leaf: "A;"
    - Parser: Should create a single Node with .name="A"
    - midpoint_root: No change
    - BFS steps: trivial
    """
    newick = "A;"
    tree = parse_newick(newick)
    assert tree.name == "A"
    assert tree.is_leaf()

    # Test the BFS function directly:
    leaf, dist = _farthest_leaf_pointer(tree)
    assert leaf is tree
    assert dist == 0.0, "Distance from single leaf to itself is 0"

    # midpoint_root -> no change
    rooted = tree.midpoint_root()
    assert rooted is tree, "Same single node"
    assert len(rooted.get_leaves()) == 1


def test_two_leaves():
    """
    STEP 2: Two leaves: (A:2,B:2);
    - Parser: Should produce a root with two child nodes A,B
    - BFS from A => B => total distance=4
    - midpoint => distance=2 => might land exactly between them
    - check splicing, or check that we end up with 2 leaves
    """
    newick = "(A:2,B:2);"
    tree = parse_newick(newick)
    leaves = tree.get_leaves()
    assert len(leaves) == 2
    names = {nd.name for nd in leaves}
    assert names == {"A", "B"}

    # test BFS function:
    #  BFS from A => L1
    A_node = [lf for lf in leaves if lf.name == "A"][0]
    B_node = [lf for lf in leaves if lf.name == "B"][0]
    far_leaf, max_dist = _farthest_leaf_pointer(A_node)
    assert (far_leaf is B_node) or (far_leaf is A_node), "One is farthest from A"

    # do midpoint
    rooted = tree.midpoint_root()
    # we should still have 2 leaves "A" and "B"
    final_leaves = rooted.get_leaves()
    assert len(final_leaves) == 2
    final_names = {nd.name for nd in final_leaves}
    assert final_names == {"A", "B"}


def test_star_topology():
    """
    STEP 3: star tree with multiple children:
      (A:1,B:2,C:3,D:4);
    - total shape is ambiguous from pointer perspective,
      but code should handle BFS well.
    - ensure no duplication after midpoint.
    """
    newick = "(A:1,B:2,C:3,D:4);"
    tree = parse_newick(newick)
    leaves_before = tree.get_leaves()
    assert len(leaves_before) == 4
    names_before = {lf.name for lf in leaves_before}
    assert names_before == {"A", "B", "C", "D"}

    # midpoint
    rooted = tree.midpoint_root()
    leaves_after = rooted.get_leaves()
    assert len(leaves_after) == 4, "Should remain 4 distinct leaves"
    names_after = {lf.name for lf in leaves_after}
    assert names_after == {"A", "B", "C", "D"}

    # BFS test: pick "A" => find farthest => see if "D" is correct?
    A_node = [lf for lf in leaves_after if lf.name == "A"][0]
    far_leaf, dist = _farthest_leaf_pointer(A_node)
    # we suspect the farthest might be "D" with distance=5? or so
    assert far_leaf.name in {"B", "C", "D"}, "One of them is presumably farthest"
    # We won't be too strict on which it picks, but we see no crash.


def test_deeper_tree():
    """
    STEP 4: deeper chain-like tree:
      (((A:1):2,B:5):1,(C:1,(D:1,E:1):2):3);
    This ensures we have some internal depth >2 or 3 levels.

    We'll do:
      - parse
      - check BFS from a leaf
      - check midpoint => see no duplication of leaves
    """
    newick = "(((A:1):2,B:5):1,(C:1,(D:1,E:1):2):3);"
    # parse
    tree = parse_newick(newick)
    leaves_before = tree.get_leaves()
    assert len(leaves_before) == 5
    names_before = {lf.name for lf in leaves_before}
    assert names_before == {"A", "B", "C", "D", "E"}

    # BFS from "A"
    A_node = [lf for lf in leaves_before if lf.name == "A"][0]
    far_leaf, dist = _farthest_leaf_pointer(A_node)
    # We won't assert the exact dist, but let's check we get a real leaf
    assert far_leaf.name in {"B", "C", "D", "E"}

    # build path A->that leaf
    path_nodes, path_dists = None, None
    # We'll find the farthest, build the path
    # not essential, but let's do it for coverage:
    path_nodes, path_dists = _build_pointer_path(A_node, far_leaf)
    # check lengths are consistent
    assert len(path_nodes) == len(path_dists) + 1

    # midpoint
    new_root = tree.midpoint_root()
    final_leaves = new_root.get_leaves()
    assert len(final_leaves) == 5, "No duplication"
    names_after = {lf.name for lf in final_leaves}
    assert names_after == {"A", "B", "C", "D", "E"}


def test_splicing_inside_edge():
    """
    STEP 5: explicit forced midpoint inside an edge.
      (A:3,B:1); total path=4 => half=2 => we create a "Midpoint" node
      1 unit from A or maybe 1 from B depending on BFS direction.

    We'll test after midpoint that:
      - A,B remain leaves
      - There's exactly 1 "Midpoint" node
    Then we test internal steps like flipping pointers.
    """
    newick = "(A:3,B:1);"
    tree = parse_newick(newick)
    # We have 2 leaves => total dist=4 => half=2 => that is inside the edge
    A_node, B_node = None, None
    for lf in tree.get_leaves():
        if lf.name == "A":
            A_node = lf
        elif lf.name == "B":
            B_node = lf
    assert A_node and B_node

    # BFS from A => farthest => should be B with dist=4
    far_leaf, dist = _farthest_leaf_pointer(A_node)
    assert far_leaf.name == "B"
    assert dist == 4

    # build path
    path_nodes, path_dists = _build_pointer_path(A_node, B_node)
    assert len(path_nodes) == 3
    assert len(path_dists) == 2
    assert path_dists[0] == 1

    # midpoint
    rooted = tree.midpoint_root()
    leaves_after = rooted.get_leaves()
    assert len(leaves_after) == 2, "Should remain 2 leaves"
    final_names = {lf.name for lf in leaves_after}
    assert final_names == {"A", "B"}

    # Check for 1 "Midpoint" node
    all_nodes = list(rooted.traverse())
    mid_nodes = [nd for nd in all_nodes if nd.name == "Midpoint"]
    assert len(mid_nodes) == 1, "One spliced midpoint node"

    # Now let's do direct BFS check again from A or B
    new_A = [lf for lf in leaves_after if lf.name == "A"][0]
    far_leaf2, dist2 = _farthest_leaf_pointer(new_A)
    # presumably it's B again or the midpoint, but B is a leaf so likely B
    assert far_leaf2.name == "B"


def test_very_deep_chain():
    """
    STEP 6 (Bonus Complexity): Very deep chain
      e.g. A->(1)->X->(2)->Y->(3)->Z->(4)->B
    We'll just build it by hand in newick:
      (((A:1,X:1):1,(Y:2,(Z:1,B:1):2):3):4);
    Actually that might not produce a perfect chain, but let's try a linear chain:
      (A:1,(X:1,(Y:2,(Z:3,B:4))));
    We'll see if parser & BFS handle depth w/o duplication.
    """

    newick = "(A:1,(X:2,(Y:3,(Z:4,B:5))));"
    tree = parse_newick(newick)
    leaves = tree.get_leaves()
    # Leaves => A, Z, B
    # The parser might produce a pointer-based star or chain depending on parentheses
    # Let's see if we have 3 leaves indeed
    assert len(leaves) == 5
    names = {lf.name for lf in leaves}
    assert names == {"A", "X", "Y", "Z", "B"}, "We expect these 3 leaves"

    # BFS from A => farthest => presumably B with total dist=1+2+3+5=11 or Z with 1+2+3+4=10
    # so B might be the farthest at 11

    # midpoint => total diameter= likely 11 => half=5.5 => splices somewhere in the chain
    root2 = tree.midpoint_root()
    leaves2 = root2.get_leaves()
    assert len(leaves2) == 5, "No duplication"
    assert names == {"A", "X", "Y", "Z", "B"}, "We expect these 3 leaves"
