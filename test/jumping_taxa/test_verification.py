import pytest
import os
import json
from pathlib import Path
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.jumping_taxa.verification import verify_jumping_taxa_solution

# Paths
TEST_TREE_DIR = Path("test/colouring/trees")


class TestJumpingTaxaVerifier:
    """Unit tests for the verify_jumping_taxa_solution function."""

    def test_verify_simple_valid_solution(self):
        """Test verification of a correct solution on simple trees."""
        # Tree 1: ((A,B),C)
        # Tree 2: ((A,C),B)
        # Pruning B (index 1) leaves (A,C) which matches ((A,C))
        tree1 = parse_newick("((A,B),C);")
        tree2 = parse_newick("((A,C),B);", encoding=tree1.taxa_encoding)
        if isinstance(tree1, list):
            tree1 = tree1[0]
        if isinstance(tree2, list):
            tree2 = tree2[0]

        candidate_taxa = ["B"]
        report = verify_jumping_taxa_solution(tree1, tree2, candidate_taxa)

        assert report["success"] is True
        assert report["metrics_before"]["tree1_leaves"] == 3
        assert report["metrics_after"]["tree1_leaves"] == 2
        assert not report["errors"]

    def test_verify_invalid_solution_topology_mismatch(self):
        """Test verification fails when pruned trees are still different."""
        # Tree 1: ((A,B),(C,D))
        # Tree 2: ((A,C),(B,D))
        # Pruning only A leaves ((B),(C,D)) vs ((C),(B,D)) -> mismatch
        tree1 = parse_newick("((A,B),(C,D));")
        tree2 = parse_newick("((A,C),(B,D));", encoding=tree1.taxa_encoding)
        if isinstance(tree1, list):
            tree1 = tree1[0]
        if isinstance(tree2, list):
            tree2 = tree2[0]

        candidate_taxa = ["A"]
        report = verify_jumping_taxa_solution(tree1, tree2, candidate_taxa)

        assert report["success"] is False
        assert (
            "Pruned trees are distinct (not isomorphic). Conflict remains."
            in report["errors"]
        )

    def test_verify_unknown_taxa_warning(self):
        """Test warning generation when candidate taxon is missing."""
        tree1 = parse_newick("((A,B),C);")
        tree2 = parse_newick("((A,B),C);", encoding=tree1.taxa_encoding)
        if isinstance(tree1, list):
            tree1 = tree1[0]
        if isinstance(tree2, list):
            tree2 = tree2[0]

        candidate_taxa = ["Z"]  # Z does not exist
        report = verify_jumping_taxa_solution(tree1, tree2, candidate_taxa)

        # Trees are already isomorphic, so success is True, but we expect a warning
        assert report["success"] is True
        assert any("not found" in w for w in report["warnings"])
        assert report["metrics_after"]["tree1_leaves"] == 3  # No deletion happened

    def test_verify_identical_trees_empty_candidate(self):
        """Test that identical trees require no jumping taxa."""
        tree1 = parse_newick("((A,B),C);")
        tree2 = parse_newick("((A,B),C);", encoding=tree1.taxa_encoding)
        if isinstance(tree1, list):
            tree1 = tree1[0]
        if isinstance(tree2, list):
            tree2 = tree2[0]

        candidate_taxa = []
        report = verify_jumping_taxa_solution(tree1, tree2, candidate_taxa)

        assert report["success"] is True
        assert not report["errors"]


# Dynamic generation of test cases from JSON files
def get_json_test_files():
    """Find all .json files in the test directory that look like test cases."""
    json_files = []
    if TEST_TREE_DIR.exists():
        for root, _, files in os.walk(TEST_TREE_DIR):
            for file in files:
                if file.endswith(".json"):
                    json_files.append(os.path.join(root, file))
    return sorted(json_files)


@pytest.mark.parametrize("json_path", get_json_test_files())
def test_verify_json_fixtures(json_path):
    """
    Run verifier on all solutions found in existing JSON fixtures.
    This effectively 'officials' the debug script into the test suite.
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        pytest.skip(f"Skipping invalid JSON: {json_path}")

    if "tree1" not in data or "tree2" not in data:
        pytest.skip(f"Skipping JSON without tree fields: {json_path}")

    if "solutions" not in data or not data["solutions"]:
        pytest.skip(f"Skipping JSON without solutions: {json_path}")

    # Parse Trees
    try:
        t1 = parse_newick(data["tree1"])
        if isinstance(t1, list):
            t1 = t1[0]
        t2 = parse_newick(data["tree2"], encoding=t1.taxa_encoding)
        if isinstance(t2, list):
            t2 = t2[0]
    except Exception as e:
        pytest.fail(f"Failed to parse trees in {json_path}: {e}")

    # Verify each solution
    solutions = data["solutions"]
    failures = []

    for i, solution_set in enumerate(solutions):
        # Flatten solution partitions into single taxa list
        candidate_taxa = []
        for partition in solution_set:
            candidate_taxa.extend(partition)

        report = verify_jumping_taxa_solution(t1, t2, candidate_taxa)

        if not report["success"]:
            failures.append(f"Solution #{i + 1} failed: {report['errors']}")

    if failures:
        # We explicitly fail the test if verification fails
        # Note: If existing fixtures are known to be broken, we might want to xfail this
        # based on a blocklist, but for now we enforce correctness.
        pytest.fail(f"Verification failed for {json_path}:\n" + "\n".join(failures))
