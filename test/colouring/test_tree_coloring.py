import json
import pytest
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.jumping_taxa import call_jumping_taxa
from brancharchitect.plot.tree_plot import plot_rectangular_tree_pair
from brancharchitect.plot.rectangular_tree import plot_tanglegram
from brancharchitect.jumping_taxa.debug import jt_logger
from brancharchitect.jumping_taxa.debug.output import (
    write_debug_output,
)
from brancharchitect.jumping_taxa.debug.error_handling import (
    log_detailed_error,
    debug_algorithm_execution,
)

# Configure logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


# Disable jumping taxa logger for test runs
jt_logger.disabled = True

# Set up output directory
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "test_debug"

# Load test data once at module level
TEST_DATA_LIST = []
tree_files = Path("./test/colouring/trees/")
for _dir in tree_files.iterdir():
    if _dir.is_dir():
        for file_path in _dir.iterdir():
            if file_path.is_file() and file_path.suffix == ".json":
                with open(file_path) as f:
                    data = json.load(f)
                    data["name"] = file_path.name
                    # Store the absolute path to the JSON file for VS Code link
                    data["source_path"] = str(file_path.absolute())
                    if "solutions" in data:
                        data["solutions"] = [
                            sorted([tuple(sorted(component)) for component in solution])
                            for solution in (data["solutions"] or [[]])
                        ]
                        TEST_DATA_LIST.append(data)


def setup_debug_output_dir():
    """Create output directory for debug files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Debug output directory created at: {OUTPUT_DIR}")
    return OUTPUT_DIR


@debug_algorithm_execution
def process_and_log_test_case(t1, t2, test_name, output_dir, source_path=None):
    """Process a test case and generate debugging output."""
    output_filename: str = f"debug_log_{os.path.splitext(test_name)[0]}.html"
    output_path = output_dir / output_filename

    try:
        jt_logger.disabled = False
        jt_logger.clear()

        jt_logger.section(f"Test Case: {test_name}")

        # Add VS Code button if source path is provided
        if source_path:
            vscode_uri = f"vscode://file/{source_path}"
            button_html = (
                f'<div style="margin: 10px 0;">'
                f'<a href="{vscode_uri}" style="display: inline-block; '
                f"background-color: #007acc; color: white; padding: 8px 16px; "
                f'text-decoration: none; border-radius: 4px; font-weight: bold;">'
                f"Open JSON Test File in VS Code</a>"
                f"</div>"
            )
            # Fixed: use html() instead of add_html()
            jt_logger.html(button_html)

        # Add visualizations
        tanglegram_svg = plot_tanglegram(t1, t2)
        jt_logger.section("Tree Tanglegram")
        jt_logger.add_svg(tanglegram_svg)

        rect_svg = plot_rectangular_tree_pair(
            t1, t2, width=800, height=400, margin=30, label_offset=2
        )
        jt_logger.section("Rectangular Tree Pair")
        jt_logger.add_svg(rect_svg)

        return call_jumping_taxa(t1, t2, algorithm="lattice")

    except Exception as e:
        log_detailed_error(
            e,
            {
                "test_file": test_name,
                "tree1": t1,
                "tree2": t2,
                "timestamp": datetime.now().isoformat(),
            },
        )
        raise

    finally:
        write_debug_output(
            output_path=str(output_path), title=f"Test Analysis: {test_name}"
        )


@pytest.fixture(
    params=TEST_DATA_LIST, ids=[test_data["name"] for test_data in TEST_DATA_LIST]
)
def test_data(request):
    return request.param


@pytest.mark.type_check()
@pytest.mark.timeout(1)
def test_tree_coloring(test_data):
    """Test the jumping taxa algorithm against known solutions."""
    output_dir = setup_debug_output_dir()

    try:
        t1 = parse_newick(test_data["tree1"])
        t2 = parse_newick(test_data["tree2"], t1._order)

        # Pass source_path to the processing function
        result = process_and_log_test_case(
            t1,
            t2,
            test_data["name"],
            output_dir,
            source_path=test_data.get("source_path"),
        )
        processed_result = sorted([tuple(sorted(pair)) for pair in result])

        translated_names = []
        rever_encoding = {v: k for k, v in t1.taxa_encoding.items()}
        for results in processed_result:
            pair = []
            for taxa in results:
                pair.append(rever_encoding[taxa])
            translated_names.append(tuple(sorted(pair)))

        sorted_translated_names = sorted(translated_names)
        jt_logger.section("Test Results")
        jt_logger.info(
            f"Test passed: {sorted_translated_names in [sorted(sol) for sol in test_data['solutions']]}"
        )
        jt_logger.info(f"Found solution: {sorted_translated_names}")
        jt_logger.info(f"Expected solutions: {test_data['solutions']}")

        if sorted_translated_names not in [
            sorted(sol) for sol in test_data["solutions"]
        ]:
            jt_logger.section("Detailed Mismatch")
            jt_logger.info(f"Got: {sorted_translated_names}")
            jt_logger.info(
                f"Expected one of: {[sorted(sol) for sol in test_data['solutions']]}"
            )
            raise AssertionError(
                f"Mismatch: got {sorted_translated_names}, expected one of {test_data['solutions']}"
            )

    except Exception as e:
        log_detailed_error(
            e,
            {
                "test_file": test_data["name"],
                "timestamp": datetime.now().isoformat(),
            },
        )
        raise
