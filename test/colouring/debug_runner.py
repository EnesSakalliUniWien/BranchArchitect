import unittest
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from brancharchitect.logger.debug import jt_logger
from test.colouring.utils import (
    discover_test_cases,
    ensure_output_dir,
    build_trees_from_data,
    process_and_log_test_case,
    translate_indices_to_names,
)


class TestTreeColoringDebug(unittest.TestCase):
    def setUp(self):
        PROJECT_ROOT = Path(__file__).parent.parent.parent
        TEST_CASES_DIR = PROJECT_ROOT / "test" / "colouring" / "trees"
        self.test_data_list = discover_test_cases(TEST_CASES_DIR)
        self.output_dir = ensure_output_dir(PROJECT_ROOT)

    def test_all_cases(self):
        print(f"Running {len(self.test_data_list)} test cases...")
        for test_data in self.test_data_list:
            case_name = test_data["name"]
            with self.subTest(case=case_name):
                print(f" Testing {case_name}...", end="", flush=True)
                try:
                    t1, t2 = build_trees_from_data(test_data)

                    result = process_and_log_test_case(
                        t1,
                        t2,
                        case_name,
                        self.output_dir,
                        source_path=test_data.get("source_path"),
                    )

                    processed_result = sorted([tuple(sorted(pair)) for pair in result])
                    translated_names = translate_indices_to_names(
                        processed_result, t1.taxa_encoding
                    )

                    expected_solutions = [sorted(sol) for sol in test_data["solutions"]]

                    if translated_names in expected_solutions:
                        print(" PASS")
                    else:
                        print(" FAIL")
                        print(f"   Got: {translated_names}")
                        print(f"   Expected one of: {expected_solutions}")
                        self.fail(f"Mismatch in {case_name}: got {translated_names}")

                except Exception as e:
                    print(f" ERROR: {e}")
                    raise e


if __name__ == "__main__":
    unittest.main()
