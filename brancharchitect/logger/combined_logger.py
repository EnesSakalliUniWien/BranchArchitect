"""Combined logger with all functionality."""

from brancharchitect.logger.base_logger import AlgorithmLogger
from brancharchitect.logger.table_logger import TableLogger
from brancharchitect.logger.matrix_logger import MatrixLogger
from brancharchitect.logger.tree_logger import TreeLogger
import logging


class Logger(TableLogger, MatrixLogger, TreeLogger):
    """
    Combined logger that inherits all visualization capabilities.

    This logger combines the functionality of:
    - TableLogger: For displaying tabular data
    - MatrixLogger: For displaying matrices with multiple representations
    - TreeLogger: For visualizing and comparing trees

    Usage:
        logger = Logger("my_algorithm")
        logger.section("Phase 1")
        logger.info("Starting phase 1...")
        logger.table(data, headers=["col1", "col2"])
        logger.matrix(matrix_data)
        logger.log_tree_comparison(tree1, tree2)
    """

    def __init__(self, name: str):
        """Initialize the combined logger."""
        # Initialize base AlgorithmLogger
        AlgorithmLogger.__init__(self, name)

    def setup_console_logging(self, level: int = logging.INFO):
        """Enable logging to the console."""
        self.disabled = False
        if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(level)
