"""Single-tree + single-alignment SatuTe integration ("Information-Scope").

Distinct from the sliding-window movie pipeline (webapp/services/trees/processing.py):
there is exactly one tree and one alignment, no windowing, and no SPR
interpolation between trees. The one thing this route adds on top of the
existing single-tree code path is running SatuTe against the processed tree
and attaching its per-branch saturation results as node annotations before
serialization, using the shared, already-generic annotation pipeline
(brancharchitect.tree.build_branch_annotation_fields) — no new wire format.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from logging import Logger
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from werkzeug.utils import secure_filename

from brancharchitect.io import parse_newick
from brancharchitect.movie_pipeline.tree_interpolation_pipeline import (
    TreeInterpolationPipeline,
)
from brancharchitect.movie_pipeline.types import PipelineConfig
from brancharchitect.satute_bridge import (
    DEFAULT_STEP,
    DEFAULT_WINDOW_SIZE,
    attach_saturation_annotations,
    compute_windows_for_branches,
    parse_sat_stat,
    run_satute,
    summarize_windows_across_branches,
    to_unrooted_newick,
)
from brancharchitect.tree import Node

DEFAULT_SATURATION_FORMULA = "eigenvalue_weighted"

# Cap on branches covered by the per-site export. Uncapped, the 1871-taxon Tree
# of Life writes ~507 MB and costs ~12 s and ~790 MB RSS to parse per formula;
# at 200 branches it is ~12 MB. Branches are kept highest-satP first, i.e. the
# ones closest to saturation and so the ones worth windowing.
SATURATION_SITES_MAX_BRANCHES = 200

from webapp.services.msa import process_msa_data
from webapp.services.trees.frontend_builder import (
    assemble_frontend_metadata,
    build_movie_data_from_result,
)


class InformationScopeError(ValueError):
    """Raised for invalid input to the Information-Scope analysis."""


ProgressCallback = Callable[[int, str], None]


def run_information_scope_analysis(
    tree_content: str,
    alignment_content: str,
    filename: str = "uploaded_file",
    alpha: float = 0.05,
    model: Optional[str] = None,
    logger: Optional[Logger] = None,
    progress_callback: Optional[ProgressCallback] = None,
    windowed: bool = True,
) -> Dict[str, Any]:
    """Parse one tree + one alignment, run SatuTe, return a full PhyloMovieData dict.

    Unlike the streaming movie endpoint, this returns a single JSON-ready dict
    (metadata plus `interpolated_trees`) — there's exactly one tree, so there's
    nothing to chunk.
    """
    def report(percent: int, message: str) -> None:
        if progress_callback:
            progress_callback(percent, message)

    filename = secure_filename(filename)

    report(2, "Parsing tree...")
    content_clean = tree_content.strip("\r")
    parsed_trees: Node | List[Node] = parse_newick(
        content_clean, treat_zero_as_epsilon=True
    )
    trees: List[Node] = (
        [parsed_trees] if isinstance(parsed_trees, Node) else parsed_trees
    )
    if len(trees) != 1:
        raise InformationScopeError(
            f"Information-Scope expects exactly one tree, found {len(trees)}."
        )

    config = PipelineConfig(
        enable_rooting=False,
        use_anchor_ordering=True,
        anchor_weight_policy="destination",
        circular=True,
        logger_name="webapp_information_scope",
    )
    pipeline = TreeInterpolationPipeline(config=config)
    result = pipeline.process_trees(trees)
    report(8, "Tree processed, preparing SatuTe run...")
    processed_root = result["interpolated_trees"][0]

    workdir = Path(tempfile.mkdtemp(prefix="information-scope-"))
    try:
        alignment_path = workdir / "alignment.fasta"
        alignment_path.write_text(alignment_content, encoding="utf-8")

        tree_path = workdir / "tree.nwk"
        tree_path.write_text(to_unrooted_newick(processed_root), encoding="utf-8")

        satute_workdir = workdir / "satute_run"
        # By far the dominant cost on large trees (~6 min for the 1871-taxon
        # Tree of Life), which is why this route streams progress rather than
        # blocking a single request.
        report(12, "Running SatuTe (this can take several minutes on large trees)...")
        run_result = run_satute(
            alignment_path,
            tree_path,
            satute_workdir,
            alpha=alpha,
            model=model,
            write_sites=windowed,
            sites_max=SATURATION_SITES_MAX_BRANCHES,
        )
        # Both formulas, so the UI can switch between them. The first is the
        # default shown before the user picks.
        report(60, "Reading branch statistics...")
        rows = parse_sat_stat(run_result.stat_file, formula=DEFAULT_SATURATION_FORMULA)
        dominant_rows = parse_sat_stat(run_result.stat_file, formula="dominant")
        if logger:
            logger.info(
                "[information_scope] SatuTe produced %d %s and %d dominant pooled rows",
                len(rows),
                DEFAULT_SATURATION_FORMULA,
                len(dominant_rows),
            )

        unmatched = attach_saturation_annotations(processed_root, rows, set_default=True)
        attach_saturation_annotations(processed_root, dominant_rows, set_default=False)
        if unmatched and logger:
            logger.warning(
                "[information_scope] %d SatuTe splits did not match any branch: %s",
                len(unmatched),
                unmatched,
            )

        # Windowed saturation. A branch can be decisively "informative" pooled
        # over the whole alignment while individual windows are saturated --
        # that contrast is the point of the track, and it is invisible in the
        # branch-level rows alone.
        saturation_windows: Dict[str, List[Dict[str, Any]]] = {}
        # How many branches the per-site export actually covered, versus how
        # many exist. Reported rather than inferred from window counts: a
        # capped run can still end up under the cap when branches drop out for
        # unusable variance, so counts alone cannot tell you it was capped.
        windowed_branch_count = 0
        if windowed:
            report(72, "Computing sliding windows...")
        if windowed and run_result.site_file is not None:
            for formula, formula_rows in (
                (DEFAULT_SATURATION_FORMULA, rows),
                ("dominant", dominant_rows),
            ):
                branch_windows = compute_windows_for_branches(
                    run_result.site_file,
                    variance_by_branch={r.branch_id: r.variance for r in formula_rows},
                    alpha=alpha,
                    formula=formula,
                )
                saturation_windows[formula] = summarize_windows_across_branches(
                    branch_windows
                )
                windowed_branch_count = max(windowed_branch_count, len(branch_windows))
            if logger:
                logger.info(
                    "[information_scope] %d sliding windows (size %d, step %d)",
                    len(saturation_windows.get(DEFAULT_SATURATION_FORMULA, [])),
                    DEFAULT_WINDOW_SIZE,
                    DEFAULT_STEP,
                )
        elif windowed and logger:
            logger.warning(
                "[information_scope] no .sat.site produced; "
                "windowed saturation unavailable (is the SatuTe build current?)"
            )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    report(88, "Processing alignment...")
    t_msa_start = time.perf_counter()
    msa_data = process_msa_data(
        msa_content=alignment_content,
        num_trees=1,
        logger=logger,
        window_size=1,
        step_size=1,
    )
    if logger:
        logger.info(
            "[information_scope] process_msa_data %.3fs", time.perf_counter() - t_msa_start
        )

    movie_data = build_movie_data_from_result(
        result=result,
        filename=filename,
        msa_data=msa_data,
    )
    report(96, "Building response...")
    metadata = assemble_frontend_metadata(movie_data)
    # "movie" must stay exactly the existing PhyloMovieData shape (the frontend's
    # validator enforces an exact key set) — anything Information-Scope-specific
    # goes as a sibling key instead, not merged into it.
    return {
        "movie": {
            **metadata,
            "interpolated_trees": movie_data.interpolated_trees,
        },
        "satute_unmatched_splits": unmatched,
        "saturation_windows": saturation_windows,
        "saturation_formulas": [DEFAULT_SATURATION_FORMULA, "dominant"],
        "saturation_default_formula": DEFAULT_SATURATION_FORMULA,
        "saturation_window_size": DEFAULT_WINDOW_SIZE,
        "saturation_windowed": windowed,
        "saturation_windowed_branch_cap": SATURATION_SITES_MAX_BRANCHES,
        "saturation_windowed_branches": windowed_branch_count,
        "saturation_total_branches": len(rows),
        "saturation_window_step": DEFAULT_STEP,
    }
