from typing import cast

from brancharchitect.logger.tree_logger import TreeLogger
from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.tree import Node


def test_log_tree_comparison_records_disabled_plotting_message() -> None:
    trees = cast(list[Node], parse_newick("((A,B),C);((A,C),B);"))
    logger = TreeLogger("tree-logger-test")

    logger.log_tree_comparison(trees[0], trees[1])

    assert "Tree plotting disabled" in logger.get_html_content()
