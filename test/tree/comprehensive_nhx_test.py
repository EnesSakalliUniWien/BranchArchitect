#!/usr/bin/env python3

from brancharchitect.parser.newick_parser import parse_newick


def test_nhx_parsing():
    """Test comprehensive NHX parsing functionality."""

    # Test 1: Simple NHX annotation
    print("=== Test 1: Simple NHX ===")
    simple_nhx = "((A:0.1,B:0.2):0.05[&&NHX:confidence=0.95],C:0.3);"
    try:
        tree = parse_newick(simple_nhx)
        print(f"✅ Parsed simple NHX tree")

        # Check if NHX metadata was extracted
        def check_nodes(node, indent=""):
            print(f"{indent}Node '{node.name}': length={node.length}")
            if hasattr(node, "values") and node.values:
                print(f"{indent}  NHX metadata: {node.values}")
            for child in node.children:
                check_nodes(child, indent + "  ")

        check_nodes(tree)

    except Exception as e:
        print(f"❌ Failed: {e}")

    print()

    # Test 2: Multiple NHX annotations
    print("=== Test 2: Multiple NHX annotations ===")
    multi_nhx = "((A:0.1[&&NHX:species=human],B:0.2[&&NHX:species=chimp]):0.05[&&NHX:bootstrap=85],C:0.3[&&NHX:species=mouse]);"
    try:
        tree = parse_newick(multi_nhx)
        print(f"✅ Parsed multiple NHX tree")

        def check_nodes(node, indent=""):
            print(f"{indent}Node '{node.name}': length={node.length}")
            if hasattr(node, "values") and node.values:
                print(f"{indent}  NHX metadata: {node.values}")
            for child in node.children:
                check_nodes(child, indent + "  ")

        check_nodes(tree)

    except Exception as e:
        print(f"❌ Failed: {e}")

    print()

    # Test 3: Original problematic case
    print("=== Test 3: Original problematic case ===")
    original_nhx = "((qfhJy6dGfL_P_serotinus_RF_complete_NCBI:0.000003[&&NHX:LWR=0.181353:LLH=-731.848948:alpha=0.000000],2FPjEKYcXz_L_noctivagans_RF_complete_NCBI:0.000013[&&NHX:LWR=0.157488:LLH=-731.848948:alpha=0.000000]):0.000013[&&NHX:LWR=0.000000:LLH=-731.848948:alpha=0.000000],x6hxPDGWKb_M_brandtii_RF_complete_NCBI:0.000057[&&NHX:LWR=0.661159:LLH=-731.848948:alpha=0.000000]);"
    try:
        tree = parse_newick(original_nhx)
        print(f"✅ Parsed original problematic NHX tree")

        def check_nodes(node, indent=""):
            print(f"{indent}Node '{node.name}': length={node.length}")
            if hasattr(node, "values") and node.values:
                print(f"{indent}  NHX metadata: {node.values}")
            for child in node.children:
                check_nodes(child, indent + "  ")

        check_nodes(tree)

    except Exception as e:
        print(f"❌ Failed: {e}")

    print()

    # Test 4: Mixed regular and NHX metadata
    print("=== Test 4: Mixed metadata formats ===")
    mixed = "((A:0.1[support=95],B:0.2[&&NHX:confidence=0.8]):0.05,C:0.3);"
    try:
        tree = parse_newick(mixed)
        print(f"✅ Parsed mixed metadata tree")

        def check_nodes(node, indent=""):
            print(f"{indent}Node '{node.name}': length={node.length}")
            if hasattr(node, "values") and node.values:
                print(f"{indent}  Metadata: {node.values}")
            for child in node.children:
                check_nodes(child, indent + "  ")

        check_nodes(tree)

    except Exception as e:
        print(f"❌ Failed: {e}")


if __name__ == "__main__":
    test_nhx_parsing()
