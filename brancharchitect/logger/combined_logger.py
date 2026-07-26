"""Combined logger with all functionality."""

from brancharchitect.logger.base_logger import AlgorithmLogger
from brancharchitect.logger.table_logger import TableLogger
from brancharchitect.logger.matrix_logger import MatrixLogger
from brancharchitect.logger.tree_logger import TreeLogger


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
