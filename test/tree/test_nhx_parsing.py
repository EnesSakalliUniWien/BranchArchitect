#!/usr/bin/env python3
"""
Test script for parsing NHX format Newick strings
"""

from brancharchitect.io import parse_newick

# Your problematic NHX string (truncated for testing)
nhx_string = """(LC769681:0.000003[&&NHX:LWR=0.181353:LLH=-731.848948:alpha=0.000000],(LC769682:0.007912[&&NHX:LWR=0.009006:LLH=-734.851523:alpha=1.000000],LC769692:0.000001[&&NHX:LWR=0.009006:LLH=-734.851479:alpha=1.000000]):0.000003[&&NHX:LWR=0.009006:LLH=-734.851523:alpha=0.000000]);"""

print("Testing NHX parsing...")
print("Input string length:", len(nhx_string))
print("\nAttempting to parse...")

try:
    trees = parse_newick(nhx_string)
    print("✅ SUCCESS! Parsed tree successfully")

    # Check if it's a single tree or list
    if isinstance(trees, list):
        tree = trees[0]
        print(f"Number of trees: {len(trees)}")
    else:
        tree = trees
        print("Single tree parsed")

    # Check for NHX metadata
    print("\nChecking for NHX metadata...")
    for node in tree.traverse():
        if hasattr(node, "values") and node.values:
            print(f"Node {node.name or 'unnamed'}: {node.values}")
            break
    else:
        print("No metadata found in nodes")

    # Get leaf names
    leaves = [node.name for node in tree.traverse() if node.is_leaf()]
    print(f"\nLeaf taxa found: {leaves}")

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback

    traceback.print_exc()
