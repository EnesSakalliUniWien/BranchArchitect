from typing import List, Dict, Any
from brancharchitect.tree import Node


def verify_jumping_taxa_solution(
    tree1: Node, tree2: Node, candidate_taxa: List[str]
) -> Dict[str, Any]:
    """
    Verify if a set of candidate jumping taxa resolves the conflict between two trees.

    This function simulates the pruning of the candidate taxa from both trees and
    checks if the resulting subtrees are topologically isomorphic.

    Args:
        tree1: The first phylogenetic tree (Node object).
        tree2: The second phylogenetic tree (Node object).
        candidate_taxa: A list of taxon names identified as "jumping taxa".

    Returns:
        A dictionary containing the verification report with the following keys:
        - "success": bool, True if the pruned trees are isomorphic.
        - "errors": List[str], a list of error messages if verification fails or input is invalid.
        - "warnings": List[str], a list of warnings (e.g., if taxa to remove were not found).
        - "metrics_before": Dict[str, int], leaf counts before pruning.
        - "metrics_after": Dict[str, int], leaf counts after pruning.
        - "details": Dict[str, Any], additional details like the encoding used.
    """
    report: Dict[str, Any] = {
        "success": False,
        "errors": [],
        "warnings": [],
        "metrics_before": {},
        "metrics_after": {},
        "details": {},
    }

    # 1. Input Validation & Deep Copy
    # We work on copies to avoid modifying the original trees in-place
    try:
        t1_copy = tree1.deep_copy()
        t2_copy = tree2.deep_copy()
    except Exception as e:
        report["errors"].append(f"Failed to copy trees: {str(e)}")
        return report

    report["metrics_before"] = {
        "tree1_leaves": len(t1_copy.get_leaves()),
        "tree2_leaves": len(t2_copy.get_leaves()),
    }

    # Store encoding for details
    report["details"]["encoding"] = t1_copy.taxa_encoding

    # Validate that candidate taxa exist in the trees
    # Note: We check against the encoding/leaves of the copies
    t1_leaves_names = {leaf.name for leaf in t1_copy.get_leaves()}
    for taxon in candidate_taxa:
        if taxon not in t1_leaves_names:
            report["warnings"].append(f"Candidate taxon '{taxon}' not found in Tree 1.")

        # We implicitly check Tree 2 assuming standard usage where trees share taxa sets,
        # but robustly we should verify against Tree 2's leaves as well if they differ.
        if taxon not in t1_copy.taxa_encoding:  # Check encoding as well for consistency
            report["warnings"].append(
                f"Candidate taxon '{taxon}' not found in Tree 1 encoding."
            )

    # 2. Pruning
    # Convert taxon names to indices for deletion
    indices_to_delete = []

    # Use the encoding from the first tree (assuming they are compatible/same universe)
    # If trees have different encodings, this logic might need adjustment, but standard
    # workflow assumes a shared encoding or compatible sets.
    encoding = t1_copy.taxa_encoding

    for taxon in candidate_taxa:
        if taxon in encoding:
            indices_to_delete.append(encoding[taxon])
        # If not in encoding, we already added a warning above.

    if not indices_to_delete:
        if candidate_taxa:
            report["errors"].append(
                "No valid indices found for provided candidate taxa."
            )
            # We proceed to check isomorphism anyway (might be trees were already same)

    try:
        if indices_to_delete:
            t1_copy.delete_taxa(indices_to_delete)
            t2_copy.delete_taxa(indices_to_delete)
    except Exception as e:
        report["errors"].append(f"Error during pruning: {str(e)}")
        return report

    report["metrics_after"] = {
        "tree1_leaves": len(t1_copy.get_leaves()),
        "tree2_leaves": len(t2_copy.get_leaves()),
    }

    # 3. Isomorphism Check
    # Node equality checks topological isomorphism (same split sets)
    try:
        is_isomorphic = t1_copy == t2_copy
    except Exception as e:
        report["errors"].append(f"Error during isomorphism check: {str(e)}")
        return report

    report["success"] = is_isomorphic

    if not is_isomorphic:
        report["errors"].append(
            "Pruned trees are distinct (not isomorphic). Conflict remains."
        )

    return report
