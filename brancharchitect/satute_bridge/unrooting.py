"""Produce an unrooted Newick representation of a BranchArchitect tree for SatuTe.

IQ-TREE's Newick reader (MTree::readTree, mtree.cpp) treats a tree as rooted
if EITHER the root node has exactly two children OR the outermost node has a
non-zero branch length — the latter is easy to trip by accident, since
Node.to_newick() always writes a length for every node including the
outermost one. SatuTe then rejects any tree flagged rooted
("SatuTe currently supports unrooted trees only", tree/satute.cpp).

This has nothing to do with the app's *display* rooting (that's a separate,
later concern for the frontend's radial layout) — it's purely about producing
Newick SatuTe's parser will accept for a tree that's otherwise unrooted
biologically (an ML tree with no meaningful root).
"""

from __future__ import annotations

from brancharchitect.tree import Node


def to_unrooted_newick(root: Node) -> str:
    """Serialize `root` as Newick with a trifurcating (or higher) top node
    and a zero-length root edge — the two things IQ-TREE's reader checks to
    decide a tree is rooted.

    If `root` already has 3+ children, only the root length needs zeroing.
    If it has exactly 2, one child's edge is also contracted into the root
    (its own children promoted, with its branch length added onto theirs).
    Always operates on a copy — the original `root` object is never mutated,
    since callers need it intact afterward for annotation and display.
    """
    working = root.deep_copy(build_split_index=False)

    if len(working.children) == 2:
        left, right = working.children

        if not left.is_leaf():
            dissolvable, sibling = left, right
        elif not right.is_leaf():
            dissolvable, sibling = right, left
        else:
            raise ValueError(
                "Cannot write an unrooted representation of a two-leaf tree "
                "(both root children are leaves)."
            )

        dissolved_length = dissolvable.length or 0.0
        for grandchild in dissolvable.children:
            grandchild.length = (grandchild.length or 0.0) + dissolved_length
            grandchild.parent = working

        working.children = [sibling] + dissolvable.children

    working.length = 0.0
    return working.to_newick()
