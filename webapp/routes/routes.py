# --------------------------------------------------------------
#  routes.py
# --------------------------------------------------------------
from __future__ import annotations

from logging import Logger
import json
from pathlib import Path
import threading
from typing import Dict, Any, Generator, Optional, Callable
from flask import Response, Flask
from flask import Blueprint, current_app, jsonify, request, send_from_directory
from brancharchitect.io import UUIDEncoder
from webapp.routes.helpers import parse_tree_data_request
from webapp.services.trees import handle_tree_content
from webapp.services.trees.processing import handle_tree_content_streaming
from webapp.services.sse import (
    format_sse_message,
    sse_response,
    channels,
    ProgressChannel,
)

from typing import Union, Tuple
import tempfile
import os
import shutil  # Added for temporary directory cleanup
from msa_to_trees.pipeline import run_pipeline, FastTreeConfig

bp = Blueprint("main", __name__)


@bp.route("/about")
def about() -> Response:
    """Simple health-check / about endpoint."""
    return jsonify(
        {"about": "Phylo-Movies API backend. See the Vue/React front-end for the UI."}
    )


@bp.route("/favicon.ico")
def favicon():
    """Serve the favicon to silence 404s in browsers."""
    return send_from_directory(
        Path(__file__).resolve().parent / "static",
        "favicon.ico",
        mimetype="image/vnd.microsoft.icon",
    )


def _run_msa_analysis_and_interpolate(
    msa_content: str,
    window_size: int,
    window_step: int,
    enable_rooting: bool,
    use_gtr: bool = True,
    use_gamma: bool = True,
    use_pseudo: bool = False,
    no_ml: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """Run the MSA → tree pipeline and return the normal response structure.

    Args:
        msa_content: The raw MSA content as a string.
        window_size: Size of the sliding window.
        window_step: Step size for the sliding window.
        enable_rooting: Whether to enable midpoint rooting.
        use_gtr: Use GTR (General Time Reversible) model for tree inference.
        use_gamma: Use gamma rate heterogeneity for tree inference.
        use_pseudo: Use pseudocounts for sequences with gaps/little overlap.
        no_ml: Disable ML NNI updates to produce strictly bifurcating trees.
        progress_callback: Optional callback for progress reporting.

    Returns:
        Dictionary with the response data.
    """

    log: Logger = current_app.logger
    temp_dir = tempfile.mkdtemp(prefix="msa-analysis-")

    def report(pct: float, msg: str) -> None:
        if progress_callback:
            progress_callback(pct, msg)

    try:
        report(0, "Starting MSA analysis...")
        log.info("[msa_analysis] Starting MSA analysis...")

        # Output directory for intermediate files (FastTree requires file I/O)
        analysis_output_dir = os.path.join(temp_dir, "output")

        # Create FastTree configuration
        fasttree_config = FastTreeConfig(
            use_gtr=use_gtr,
            use_gamma=use_gamma,
            use_pseudo=use_pseudo,
            no_ml=no_ml,
        )

        report(
            20,
            f"Running tree inference pipeline with {fasttree_config.description} model...",
        )

        log.info(
            f"[msa_analysis] Running analysis pipeline with {fasttree_config.description} model..."
        )

        # Pass MSA content directly - no need to write input file to disk
        pipeline_result = run_pipeline(
            input_file=None,
            output_directory=analysis_output_dir,
            window_size=window_size,
            step_size=window_step,
            fasttree_config=fasttree_config,
            msa_content=msa_content,  # In-memory content for webservice
        )

        tree_file_path = pipeline_result.tree_file_path

        # Log dropped taxa information
        if pipeline_result.has_dropped_taxa:
            log.warning(
                f"[msa_analysis] {len(pipeline_result.dropped_taxa)} taxa were dropped "
                f"due to invalid data in some windows: {pipeline_result.dropped_taxa[:5]}..."
            )
            report(
                30,
                f"Note: {len(pipeline_result.dropped_taxa)} taxa dropped (gaps/ambiguous in some windows)",
            )

        if not os.path.exists(tree_file_path):
            raise FileNotFoundError(
                "Analysis script finished but did not produce the expected tree file."
            )

        report(60, "Processing generated trees...")

        with open(tree_file_path, "r", encoding="utf-8") as f:
            tree_content = f.read()

        log.info(
            f"[msa_analysis] Generated {pipeline_result.num_windows} trees with "
            f"{pipeline_result.kept_taxa} taxa. Running interpolation service."
        )

        # Create a sub-callback that maps 60-100 range
        tree_progress_callback = _create_sub_progress_callback(
            progress_callback, 60, 100
        )

        response_data = handle_tree_content(
            tree_content,
            filename=os.path.basename(str(tree_file_path)),
            msa_content=msa_content,
            enable_rooting=enable_rooting,
            window_size=window_size,
            window_step=window_step,
            progress_callback=tree_progress_callback,
        )

        # Add pipeline feedback about dropped taxa to response
        response_data["pipeline_info"] = {
            "total_taxa_in_alignment": pipeline_result.total_taxa,
            "taxa_in_trees": pipeline_result.kept_taxa,
            "dropped_taxa": pipeline_result.dropped_taxa,
            "dropped_taxa_count": len(pipeline_result.dropped_taxa),
            "num_trees_generated": pipeline_result.num_windows,
            "model_used": fasttree_config.description,
        }

        if pipeline_result.has_dropped_taxa:
            response_data["warnings"] = response_data.get("warnings", [])
            response_data["warnings"].append(
                f"{len(pipeline_result.dropped_taxa)} taxa were excluded because they "
                f"had only gaps or ambiguous characters in some alignment windows. "
                f"This ensures all trees have consistent taxa."
            )

        return response_data
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        log.info(f"[msa_analysis] Cleaned up temporary directory: {temp_dir}")


# ----------------------------------------------------------------------
# Main business endpoint – tree upload & analysis
# ----------------------------------------------------------------------


@bp.route("/treedata", methods=["POST"])
def treedata() -> Union[Response, Tuple[dict[str, Any], int]]:
    log: Logger = current_app.logger
    log.info("[treedata] POST /treedata from %s", request.remote_addr)

    try:
        req_data = parse_tree_data_request(request)

        # --- Handle MSA-only input synchronously so response shape matches tree uploads ---
        if req_data.tree_content is None:
            if not req_data.msa_content:
                raise ValueError("Uploaded file 'msaFile' is empty.")

            log.info(
                "[treedata] MSA file provided without a tree file. Running analysis inline for consistent response."
            )

            response_data = _run_msa_analysis_and_interpolate(
                msa_content=req_data.msa_content,
                window_size=req_data.window_size,
                window_step=req_data.window_step,
                enable_rooting=req_data.enable_rooting,
                use_gtr=req_data.use_gtr,
                use_gamma=req_data.use_gamma,
                use_pseudo=req_data.use_pseudo,
                no_ml=req_data.no_ml,
            )

            log.info(
                f"[treedata] Processed {len(response_data.get('interpolated_trees', []))} trees from MSA-only request"
            )

            return Response(
                json.dumps(response_data, cls=UUIDEncoder),
                mimetype="application/json",
            )

        # --- Existing logic for tree file upload ---
        log.info(
            f"[treedata] Processing file: {req_data.tree_filename}, "
            f"Rooting: {req_data.enable_rooting}, "
            f"Window: {req_data.window_size}, Step: {req_data.window_step}, "
            f"MSA provided: {req_data.msa_content is not None}"
        )

        response_data: Dict[str, Any] = handle_tree_content(
            req_data.tree_content,
            filename=req_data.tree_filename or "uploaded_file",
            msa_content=req_data.msa_content,
            enable_rooting=req_data.enable_rooting,
            window_size=req_data.window_size,
            window_step=req_data.window_step,
        )

        log.info(
            f"[treedata] Processed {len(response_data.get('interpolated_trees', []))} trees"
        )

        return Response(
            json.dumps(response_data, cls=UUIDEncoder), mimetype="application/json"
        )

    except ValueError as e:
        log.warning(f"[treedata] Bad request: {e}")
        log.warning("[treedata] Full traceback:", exc_info=True)
        return _fail(400, str(e)), 400

    except Exception as e:
        log.error("[treedata] Exception: %s", str(e), exc_info=True)
        return _fail(500, str(e)), 500


# ----------------------------------------------------------------------
# Streaming tree processing endpoint
# ----------------------------------------------------------------------


@bp.route("/treedata/stream", methods=["POST"])
def treedata_stream() -> Union[Response, Tuple[dict[str, Any], int]]:
    """
    Streaming version of /treedata that returns a channel_id immediately.

    Progress updates are sent via SSE on /stream/progress/<channel_id>.
    Final result is sent as a 'complete' event with the full response data.
    """
    log: Logger = current_app.logger
    log.info("[treedata/stream] POST /treedata/stream from %s", request.remote_addr)

    try:
        req_data = parse_tree_data_request(request)
        channel = channels.create()

        # Capture Flask app for background thread
        # Use getattr to avoid Pylance warning about protected attribute
        app: Flask = getattr(current_app, "_get_current_object")()

        def process_in_background() -> None:
            """Run tree processing in background thread with progress updates."""
            with app.app_context():
                try:
                    channel.send_progress(0, "Starting processing...")

                    # Handle MSA-only input
                    if req_data.tree_content is None:
                        if not req_data.msa_content:
                            channel.complete(error="Uploaded file 'msaFile' is empty.")
                            return

                        channel.send_progress(10, "Running MSA analysis...")
                        response_data = _run_msa_analysis_and_interpolate(
                            msa_content=req_data.msa_content,
                            window_size=req_data.window_size,
                            window_step=req_data.window_step,
                            enable_rooting=req_data.enable_rooting,
                            use_gtr=req_data.use_gtr,
                            use_gamma=req_data.use_gamma,
                            use_pseudo=req_data.use_pseudo,
                            no_ml=req_data.no_ml,
                            progress_callback=_make_progress_callback(channel, 10, 85),
                        )

                        # Extract trees from response for chunked streaming
                        trees = response_data.pop("interpolated_trees", [])
                        metadata = response_data  # Everything except trees

                        # Always use chunked streaming to avoid SSE message size limits
                        log.info(
                            f"[treedata/stream] MSA analysis: Streaming {len(trees)} trees in chunks"
                        )
                        channel.send_progress(90, "Streaming trees...")
                        channel.send({"metadata": metadata}, event="metadata")
                        channel.send_trees_chunked(trees, chunk_size=100)
                        channel.send_progress(100, "Complete")
                        channel.complete(
                            data={"chunked": True, "tree_count": len(trees)}
                        )
                    else:
                        # Tree file processing with chunked streaming for large datasets
                        channel.send_progress(10, "Parsing tree file...")
                        metadata, trees, use_chunked = handle_tree_content_streaming(
                            req_data.tree_content,
                            filename=req_data.tree_filename or "uploaded_file",
                            msa_content=req_data.msa_content,
                            enable_rooting=req_data.enable_rooting,
                            window_size=req_data.window_size,
                            window_step=req_data.window_step,
                            progress_callback=_make_progress_callback(channel, 10, 85),
                        )

                        # Always use chunked streaming to avoid SSE message size limits
                        log.info(
                            f"[treedata/stream] Streaming {len(trees)} trees in chunks"
                        )
                        channel.send_progress(90, "Streaming trees...")
                        channel.send({"metadata": metadata}, event="metadata")
                        channel.send_trees_chunked(trees, chunk_size=100)
                        channel.send_progress(100, "Complete")
                        channel.complete(
                            data={"chunked": True, "tree_count": len(trees)}
                        )

                except Exception as e:
                    log.error(
                        "[treedata/stream] Processing error: %s", str(e), exc_info=True
                    )
                    channel.complete(error=str(e))

        # Start background processing
        thread = threading.Thread(target=process_in_background, daemon=True)
        thread.start()

        return jsonify({"channel_id": channel.channel_id})

    except ValueError as e:
        log.warning(f"[treedata/stream] Bad request: {e}")
        return _fail(400, str(e)), 400

    except Exception as e:
        log.error("[treedata/stream] Exception: %s", str(e), exc_info=True)
        return _fail(500, str(e)), 500


def _make_progress_callback(
    channel: ProgressChannel, start_pct: int, end_pct: int
) -> Callable[[float, str], None]:
    """
    Create a progress callback that maps 0-100 input to start_pct-end_pct range.

    Args:
        channel: The progress channel to send updates to.
        start_pct: Starting percentage (e.g., 10).
        end_pct: Ending percentage (e.g., 90).

    Returns:
        Callback function that accepts (progress: float, message: str).
    """

    def callback(progress: float, message: str = "") -> None:
        # Map 0-100 to start_pct-end_pct range
        mapped = start_pct + (progress / 100.0) * (end_pct - start_pct)
        channel.send_progress(int(mapped), message)

    return callback


# ----------------------------------------------------------------------
# Diagnostic helpers
# ----------------------------------------------------------------------
@bp.route("/cause-error")
def cause_error():  # noqa: D401 – test helper does not need docstring galore
    raise Exception("Intentional test error - handled by global error handler")


@bp.errorhandler(Exception)
def global_error(exc: Exception):  # Flask passes the exception instance in
    current_app.logger.error("[global] Unhandled exception", exc_info=True)
    return _fail(500, str(exc)), 500


# ----------------------------------------------------------------------
# Utility: short error JSON helper
# ----------------------------------------------------------------------
def _fail(status_code: int, message: str) -> dict[str, Any]:
    return {
        "error": message,
        "status": status_code,
    }


def _create_sub_progress_callback(
    parent_callback: Optional[Callable[[float, str], None]],
    start_pct: float,
    end_pct: float,
) -> Optional[Callable[[float, str], None]]:
    """
    Create a sub-progress callback that maps 0-100 to a sub-range.

    Args:
        parent_callback: The parent callback to delegate to (can be None).
        start_pct: Starting percentage in parent range.
        end_pct: Ending percentage in parent range.

    Returns:
        A callback that maps progress to the sub-range, or None if parent is None.
    """
    if parent_callback is None:
        return None

    def callback(pct: float, msg: str) -> None:
        mapped = start_pct + (pct / 100.0) * (end_pct - start_pct)
        parent_callback(mapped, msg)

    return callback


# ----------------------------------------------------------------------
# SSE Streaming Endpoints
# ----------------------------------------------------------------------


@bp.route("/stream/progress/<channel_id>")
def stream_progress(channel_id: str) -> Response:
    """
    SSE endpoint to stream progress updates for a processing task.

    Connect to this endpoint after initiating a task that returns a channel_id.
    The stream will emit events: 'progress', 'log', 'error', 'complete'.

    Example client usage (JavaScript):
        const eventSource = new EventSource(`/stream/progress/${channelId}`);
        eventSource.addEventListener('progress', (e) => {
            const data = JSON.parse(e.data);
            console.log(`Progress: ${data.percent}%`);
        });
        eventSource.addEventListener('complete', (e) => {
            eventSource.close();
        });
    """
    log: Logger = current_app.logger
    channel = channels.get(channel_id)

    if channel is None:
        log.warning(f"[stream] Channel not found: {channel_id}")
        return Response(
            format_sse_message({"error": "Channel not found"}, event="error"),
            mimetype="text/event-stream",
            status=404,
        )

    log.info(f"[stream] Client connected to channel: {channel_id}")
    return sse_response(channel.stream())


@bp.route("/stream/test")
def stream_test() -> Response:
    """
    Test SSE endpoint that streams numbered messages.

    Useful for verifying SSE connectivity.
    """
    import time

    def generate() -> Generator[str, None, None]:
        for i in range(10):
            yield format_sse_message(
                {"count": i + 1, "message": f"Test message {i + 1}"},
                event="test",
            )
            time.sleep(0.5)
        yield format_sse_message({"status": "complete"}, event="complete")

    return sse_response(generate())
