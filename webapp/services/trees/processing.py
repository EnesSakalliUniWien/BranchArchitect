"""
Core tree processing functionality.
"""

from logging import Logger
from typing import List, Optional, Dict, Any, Callable, Tuple

from flask import current_app
from werkzeug.utils import secure_filename

from brancharchitect.io import parse_newick
from brancharchitect.movie_pipeline.tree_interpolation_pipeline import (
    TreeInterpolationPipeline,
)
from brancharchitect.movie_pipeline.types import PipelineConfig
from brancharchitect.tree import Node

from webapp.services.trees.frontend_builder import (
    assemble_frontend_metadata,
    build_movie_data_from_result,
    create_empty_movie_data,
)

# Type alias for progress callback
ProgressCallback = Callable[[float, str], None]


def _sub_progress(
    parent: Optional[ProgressCallback], start: float, end: float
) -> Optional[ProgressCallback]:
    """Create a sub-callback that maps 0-100 into [start, end] on the parent."""
    if parent is None:
        return None

    def callback(pct: float, msg: str) -> None:
        mapped = start + (pct / 100.0) * (end - start)
        parent(mapped, msg)

    return callback


def handle_tree_content_streaming(
    tree_content: str,
    filename: str = "uploaded_file",
    msa_content: Optional[str] = None,
    enable_rooting: bool = False,
    window_size: int = 1,
    window_step: int = 1,
    progress_callback: Optional[ProgressCallback] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Process tree content and return metadata separately from trees.

    Use this for chunked streaming to avoid sending massive JSON payloads.

    Returns:
        Tuple of:
        - metadata: Dict with all movie fields except trees
        - trees: List of serialized tree dicts
    """
    from webapp.services.msa import process_msa_data

    def report(pct: float, msg: str) -> None:
        if progress_callback:
            progress_callback(pct, msg)

    filename = secure_filename(filename)
    logger: Logger = current_app.logger
    logger.info(f"Processing uploaded file (streaming): {filename}")

    report(0, "Parsing tree file...")
    content_clean = tree_content.strip("\r")
    parsed_trees: Node | List[Node] = parse_newick(
        content_clean, treat_zero_as_epsilon=True
    )

    trees: List[Node] = (
        [parsed_trees] if isinstance(parsed_trees, Node) else parsed_trees
    )

    if not trees:
        logger.debug("No trees parsed - returning empty response")
        empty_movie_data = create_empty_movie_data(filename)
        return assemble_frontend_metadata(empty_movie_data), []

    logger.info(f"Successfully parsed {len(trees)} trees")
    report(10, f"Parsed {len(trees)} trees, computing interpolation...")

    config = PipelineConfig(
        enable_rooting=enable_rooting,
        use_anchor_ordering=True,
        anchor_weight_policy="destination",
        circular=True,
        logger_name="webapp_pipeline",
    )

    pipeline = TreeInterpolationPipeline(config=config)
    # Map pipeline's 0-100 progress into the 10-75 range so it doesn't
    # conflict with the surrounding report() calls.
    pipeline_callback = _sub_progress(progress_callback, 10, 75)
    result = pipeline.process_trees(trees, progress_callback=pipeline_callback)

    report(75, "Processing MSA data...")

    msa_data = process_msa_data(
        msa_content=msa_content,
        num_trees=len(trees),
        logger=logger,
        window_size=window_size,
        step_size=window_step,
    )

    report(90, "Building response...")

    sorted_leaves: List[str] = []
    if result["interpolated_trees"]:
        first_tree = result["interpolated_trees"][0]
        encoding = first_tree.taxa_encoding or {}
        sorted_leaves = [
            name for name, _ in sorted(encoding.items(), key=lambda item: item[1])
        ]

    movie_data = build_movie_data_from_result(
        result=result,
        filename=filename,
        msa_data=msa_data,
        sorted_leaves=sorted_leaves,
    )

    metadata = assemble_frontend_metadata(movie_data)
    serialized_trees = movie_data.interpolated_trees

    report(100, "Complete")

    return metadata, serialized_trees
