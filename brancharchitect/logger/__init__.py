"""Logging package for BranchArchitect."""

from brancharchitect.logger.base_logger import AlgorithmLogger
from brancharchitect.logger.table_logger import TableLogger
from brancharchitect.logger.matrix_logger import MatrixLogger
from brancharchitect.logger.tree_logger import TreeLogger
from brancharchitect.logger.combined_logger import Logger
from brancharchitect.logger.formatting import (
    format_set,
    beautify_frozenset,
    format_partition,
    format_partition_set,
)

# Unified singleton for algortihm visualization
jt_logger = Logger("JumpingTaxa")
jt_logger.disabled = True

__all__ = [
    "AlgorithmLogger",
    "TableLogger",
    "MatrixLogger",
    "TreeLogger",
    "Logger",
    "jt_logger",
    "format_set",
    "beautify_frozenset",
    "format_partition",
    "format_partition_set",
]
