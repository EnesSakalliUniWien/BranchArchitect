import os
import logging
from pathlib import Path

from brancharchitect.jumping_taxa.debug import jt_logger


def pytest_configure(config):
    """Set up test environment before tests run."""
    # Create debug output directory
    output_dir = (
        Path(os.path.dirname(os.path.dirname(__file__))) / "output" / "test_debug"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Enable jumping taxa logger
    jt_logger.disabled = False


def pytest_sessionfinish(session, exitstatus):
    """Clean up after tests complete."""
    # Write out any pending debug logs
    pass
