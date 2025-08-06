# Test for depth assignment in Newick parsing
from brancharchitect.parser.newick_parser import parse_newick


def collect_depths(node, result=None):
    if result is None:
        result = []
    result.append((node.name, node.depth))
    for child in getattr(node, "children", []):
        collect_depths(child, result)
    return result


def test_pytest_discovery():
    assert 1 == 1


def test_newick_depth_assignment():
    # Simple tree: ((A,B),C);
    newick = "((A,B),C);"
    tree = parse_newick(newick)
    depths = dict(collect_depths(tree))
    # The root node is a dummy node named 'root', its child is the actual root
    actual_root = (
        tree.children[0] if hasattr(tree, "children") and tree.children else tree
    )
    assert actual_root.depth == 1
    print("Actual root name:", actual_root.name, "depth:", actual_root.depth)
    print("Children of actual root:")
    for idx, child in enumerate(actual_root.children):
        print(
            f"  Child {idx}: name={child.name!r}, depth={child.depth}, children={[c.name for c in getattr(child, 'children', [])]}"
        )
        for gidx, grandchild in enumerate(getattr(child, "children", [])):
            print(
                f"    Grandchild {gidx}: name={grandchild.name!r}, depth={grandchild.depth}"
            )

    # Try a simpler tree: (A,B,C);
    print("\nTesting flat tree (A,B,C);")
    flat_tree = parse_newick("(A,B,C);")
    flat_root = (
        flat_tree.children[0]
        if hasattr(flat_tree, "children") and flat_tree.children
        else flat_tree
    )
    print("Flat root name:", flat_root.name, "depth:", flat_root.depth)
    for idx, child in enumerate(flat_root.children):
        print(f"  Child {idx}: name={child.name!r}, depth={child.depth}")
