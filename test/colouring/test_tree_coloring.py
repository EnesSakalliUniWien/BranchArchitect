import pytest
from pathlib import Path
from brancharchitect.logger.debug import jt_logger
from .utils import (
    discover_test_cases,
    ensure_output_dir,
    build_trees_from_data,
    process_and_log_test_case,
    translate_indices_to_names,
)


# Discover test cases once at module import
PROJECT_ROOT = Path(__file__).parent.parent.parent
TEST_CASES_DIR = PROJECT_ROOT / "test" / "colouring" / "trees"
TEST_DATA_LIST = discover_test_cases(TEST_CASES_DIR)


@pytest.fixture(params=TEST_DATA_LIST, ids=[td["name"] for td in TEST_DATA_LIST])
def test_data(request):
    return request.param


@pytest.mark.type_check()
@pytest.mark.timeout(10)
def test_tree_coloring(test_data):
    """Test the jumping taxa algorithm against known solutions."""
    output_dir = ensure_output_dir(PROJECT_ROOT)

    # Build trees
    t1, t2 = build_trees_from_data(test_data)

    # Run and log
    result = process_and_log_test_case(
        t1,
        t2,
        test_data["name"],
        output_dir,
        source_path=test_data.get("source_path"),
    )

    # Normalize result: indices -> names
    processed_result = sorted([tuple(sorted(pair)) for pair in result])
    translated_names = translate_indices_to_names(processed_result, t1.taxa_encoding)

    jt_logger.section("Test Results")
    jt_logger.info(
        f"Test passed: {translated_names in [sorted(sol) for sol in test_data['solutions']]}"
    )
    jt_logger.info(f"Found solution: {translated_names}")
    jt_logger.info(f"Expected solutions: {test_data['solutions']}")

    if translated_names not in [sorted(sol) for sol in test_data["solutions"]]:
        jt_logger.section("Detailed Mismatch")
        jt_logger.info(f"Got: {translated_names}")
        jt_logger.info(
            f"Expected one of: {[sorted(sol) for sol in test_data['solutions']]}"
        )
        raise AssertionError(
            f"Mismatch: got {translated_names}, expected one of {test_data['solutions']}"
        )
