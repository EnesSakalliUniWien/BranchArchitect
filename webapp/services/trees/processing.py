"""
Core tree processing functionality.
"""

from logging import Logger
from typing import List, Optional, Dict, Any

from flask import current_app
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from brancharchitect.io import parse_newick
from brancharchitect.movie_pipeline.tree_interpolation_pipeline import (
    TreeInterpolationPipeline,
)
from brancharchitect.movie_pipeline.types import InterpolationResult, PipelineConfig
from brancharchitect.tree import Node

from webapp.services.trees.frontend_builder import (
    assemble_frontend_dict,
    build_movie_data_from_result,
    create_empty_movie_data,
)


def handle_uploaded_file(
    file_storage: FileStorage,
    msa_content: Optional[str] = None,
    enable_rooting: bool = False,
    window_size: int = 1,
    window_step: int = 1,
) -> Dict[str, Any]:
    """
    Process an uploaded Newick file and compute visualization data.

    Args:
        file_storage: The uploaded tree file.
        msa_content: Optional MSA content for window inference.
        enable_rooting: Whether to enable midpoint rooting.
        window_size: Window size for tree processing.
        window_step: Window step size for tree processing.

    Returns:
        Dictionary containing all data needed for front-end visualization.
    """
    filename_value = file_storage.filename or "uploaded_file"
    filename = secure_filename(filename_value)
    logger: Logger = current_app.logger
    logger.info(f"Processing uploaded file: {filename}")

    content = file_storage.read().decode("utf-8").strip("\r")
    parsed_trees: Node | List[Node] = parse_newick(content, treat_zero_as_epsilon=True)

    # Ensure trees is always a list
    trees: List[Node] = (
        [parsed_trees] if isinstance(parsed_trees, Node) else parsed_trees
    )

    if not trees:
        logger.debug("No trees parsed - returning empty response")
        return _create_empty_response(filename)

    logger.info(f"Successfully parsed {len(trees)} trees")

    # Process trees through the pipeline
    config = PipelineConfig(
        enable_rooting=enable_rooting,
        use_anchor_ordering=True,
        anchor_weight_policy="destination",
        circular=True,
        logger_name="webapp_pipeline",
    )

    pipeline = TreeInterpolationPipeline(config=config)
    processed_data: InterpolationResult = pipeline.process_trees(trees=trees)

    # Process MSA data if available
    from webapp.services.msa import process_msa_data

    msa_data = process_msa_data(
        msa_content=msa_content,
        num_trees=len(trees),
        logger=logger,
        window_size=window_size,
        step_size=window_step,
    )

    return _create_structured_response(
        processed_data, filename, msa_data, enable_rooting, logger
    )


def _create_structured_response(
    result: InterpolationResult,
    filename: str,
    msa_data: Dict[str, Any],
    enable_rooting: bool,
    logger: Logger,
) -> Dict[str, Any]:
    """
    Create hierarchical API response using MovieData class.

    Args:
        result: InterpolationResult from TreeInterpolationPipeline.
        filename: Original filename.
        msa_data: Processed MSA data.
        enable_rooting: Whether rooting was enabled.
        logger: Logger instance.

    Returns:
        Hierarchical dictionary for API response.
    """
    # Extract leaf names from the first tree
    sorted_leaves: List[str] = []
    if result["interpolated_trees"]:
        first_tree = result["interpolated_trees"][0]

        if hasattr(first_tree, "leaves"):
            sorted_leaves = [leaf.name for leaf in first_tree.leaves]
        elif hasattr(first_tree, "get_leaves"):
            sorted_leaves = [leaf.name for leaf in first_tree.get_leaves()]
        else:
            logger.warning(
                f"Unexpected tree type: {type(first_tree)}, using fallback leaf extraction"
            )
            sorted_leaves = []

    movie_data = build_movie_data_from_result(
        result=result,
        filename=filename,
        msa_data=msa_data,
        enable_rooting=enable_rooting,
        sorted_leaves=sorted_leaves,
    )
    return assemble_frontend_dict(movie_data)


def _create_empty_response(filename: str) -> Dict[str, Any]:
    """Create an empty hierarchical response for failed processing."""
    empty_movie_data = create_empty_movie_data(filename)
    return assemble_frontend_dict(empty_movie_data)
