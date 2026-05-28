"""
Core tree processing functionality.
"""

from collections.abc import Iterable
from collections import Counter
from logging import Logger
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeAlias

from flask import current_app
from werkzeug.utils import secure_filename

from brancharchitect.io import parse_newick
from brancharchitect.movie_pipeline.tree_interpolation_pipeline import (
    TreeInterpolationPipeline,
)
from brancharchitect.movie_pipeline.types import PipelineConfig
from brancharchitect.tree import Node, build_branch_annotation_fields

from webapp.services.trees.frontend_builder import (
    assemble_frontend_metadata,
    build_movie_data_from_result,
    create_empty_movie_data,
)

# Type alias for progress callback
ProgressCallback = Callable[[float, str], None]
IndexLike: TypeAlias = int | str | bytes
_IQTREE_SINGLE_VALUE_SUPPORT_MODES = {"ufboot", "sh_alrt"}
_TREE_SERIES_SUPPORT_KIND = "bootstrap_replicate_split_frequency"


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


def _is_single_numeric_internal_label(node: Node) -> bool:
    internal_label = (node.name or "").strip()
    if not internal_label or "/" in internal_label:
        return False
    try:
        float(internal_label)
    except ValueError:
        return False
    return True


def _annotate_iqtree_single_value_support(
    trees: List[Node],
    iqtree_support_mode: Optional[str],
) -> None:
    if iqtree_support_mode not in _IQTREE_SINGLE_VALUE_SUPPORT_MODES:
        return

    for tree in trees:
        for node in tree.traverse():
            if node.is_leaf() or not _is_single_numeric_internal_label(node):
                continue
            node.values["support_kind"] = iqtree_support_mode


def _normalize_indices(indices: Optional[Iterable[IndexLike]]) -> tuple[int, ...]:
    if indices is None:
        return ()
    return tuple(sorted({int(index) for index in indices}))


def _canonical_split_key(
    split_indices: Optional[Iterable[IndexLike]],
    all_taxa_indices: tuple[int, ...],
) -> tuple[int, ...]:
    split = _normalize_indices(split_indices)
    if not split or not all_taxa_indices or split == all_taxa_indices:
        return ()

    split_set = set(split)
    complement = tuple(index for index in all_taxa_indices if index not in split_set)
    if not complement:
        return ()
    if len(split) < len(complement):
        return split
    if len(complement) < len(split):
        return complement
    return min(split, complement)


def _tree_split_keys(
    tree: Node, all_taxa_indices: tuple[int, ...]
) -> set[tuple[int, ...]]:
    keys: set[tuple[int, ...]] = set()
    for node in tree.traverse():
        if node.is_leaf():
            continue
        key = _canonical_split_key(node.split_indices.indices, all_taxa_indices)
        if key:
            keys.add(key)
    return keys


def _has_branch_support_annotation(node: Node) -> bool:
    fields = build_branch_annotation_fields(node)
    return any(field.get("role") == "branch_support" for field in fields.values())


def _format_support_label(value: float) -> str:
    return f"{value:.6g}"


def _annotate_tree_series_split_frequency(trees: List[Node]) -> None:
    if len(trees) < 2:
        return

    all_taxa_indices = _normalize_indices(trees[0].split_indices.indices)
    if not all_taxa_indices:
        return

    replicate_total = len(trees)
    counts: Counter[tuple[int, ...]] = Counter()
    for tree in trees:
        if _normalize_indices(tree.split_indices.indices) != all_taxa_indices:
            return
        counts.update(_tree_split_keys(tree, all_taxa_indices))

    for tree in trees:
        for node in tree.traverse():
            if node.is_leaf() or _has_branch_support_annotation(node):
                continue

            key = _canonical_split_key(node.split_indices.indices, all_taxa_indices)
            replicate_count = counts.get(key, 0)
            if not key or replicate_count <= 0:
                continue

            support_percent = 100 * replicate_count / replicate_total
            node.values.update(
                {
                    "support_kind": _TREE_SERIES_SUPPORT_KIND,
                    "bootstrap_frequency": _format_support_label(support_percent),
                    "replicate_count": replicate_count,
                    "replicate_total": replicate_total,
                }
            )


def handle_tree_content_streaming(
    tree_content: str,
    filename: str = "uploaded_file",
    msa_content: Optional[str] = None,
    enable_rooting: bool = False,
    window_size: int = 1,
    window_step: int = 1,
    iqtree_support_mode: Optional[str] = None,
    annotate_tree_series_support: bool = False,
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
    _annotate_iqtree_single_value_support(trees, iqtree_support_mode)
    if annotate_tree_series_support:
        _annotate_tree_series_split_frequency(trees)

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

    movie_data = build_movie_data_from_result(
        result=result,
        filename=filename,
        msa_data=msa_data,
    )

    metadata = assemble_frontend_metadata(movie_data)
    serialized_trees = movie_data.interpolated_trees

    report(100, "Complete")

    return metadata, serialized_trees
