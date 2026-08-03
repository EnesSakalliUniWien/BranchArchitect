# --------------------------------------------------------------
#  routes.py
# --------------------------------------------------------------
from __future__ import annotations

from importlib import metadata
from logging import Logger
from pathlib import Path
import threading
from typing import Dict, Any, Generator, Optional, Callable
from flask import Response, Flask
from flask import Blueprint, current_app, jsonify, request, send_from_directory
from webapp.routes.helpers import parse_tree_data_request
from webapp.services.trees.processing import handle_tree_content_streaming
from webapp.services.trees.stream_contract import send_movie_stream
from webapp.services.sse import (
    format_sse_message,
    sse_response,
    channels,
    ProgressChannel,
)

from typing import Union, Tuple
import tempfile
import os
import shutil
from msa_to_trees.pipeline import run_pipeline, FastTreeConfig, IQTreeConfig

bp = Blueprint("main", __name__)
_MSA_TREE_INFERENCE_LOCK = threading.Lock()
_MSA_TREE_INFERENCE_ACTIVE_CHANNEL_ID: str | None = None
_MSA_TREE_INFERENCE_STATE_LOCK = threading.Lock()


def _is_msa_tree_inference_request(req_data: Any) -> bool:
    return req_data.tree_content is None and bool(req_data.msa_content)


def _claim_msa_tree_inference_slot(channel_id: str) -> bool:
    global _MSA_TREE_INFERENCE_ACTIVE_CHANNEL_ID

    if not _MSA_TREE_INFERENCE_LOCK.acquire(blocking=False):
        return False

    with _MSA_TREE_INFERENCE_STATE_LOCK:
        _MSA_TREE_INFERENCE_ACTIVE_CHANNEL_ID = channel_id
    return True


def _release_msa_tree_inference_slot(channel_id: str) -> None:
    global _MSA_TREE_INFERENCE_ACTIVE_CHANNEL_ID

    with _MSA_TREE_INFERENCE_STATE_LOCK:
        if _MSA_TREE_INFERENCE_ACTIVE_CHANNEL_ID != channel_id:
            return
        _MSA_TREE_INFERENCE_ACTIVE_CHANNEL_ID = None
    _MSA_TREE_INFERENCE_LOCK.release()


def _get_msa_tree_inference_status() -> dict[str, Any]:
    with _MSA_TREE_INFERENCE_STATE_LOCK:
        active_channel_id = _MSA_TREE_INFERENCE_ACTIVE_CHANNEL_ID
    return {
        "busy": active_channel_id is not None,
        "active_channel_id": active_channel_id,
    }


@bp.route("/")
def index() -> Response:
    """Serve a small backend landing page for direct browser visits."""
    return Response(
        """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Phylo-Movies Backend</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2rem; line-height: 1.5; }
    code { background: #f3f4f6; padding: 0.125rem 0.25rem; border-radius: 0.25rem; }
  </style>
</head>
<body>
  <h1>Phylo-Movies Backend</h1>
  <p>The API server is running on <code>127.0.0.1:5002</code>.</p>
  <p>Open the frontend at <a href="http://127.0.0.1:5173/">http://127.0.0.1:5173/</a>.</p>
  <p>Backend readiness: <a href="/health">/health</a></p>
</body>
</html>
""",
        mimetype="text/html",
    )


@bp.route("/about")
def about() -> Response:
    """Simple health-check / about endpoint."""
    return jsonify(
        {"about": "Phylo-Movies API backend. See the React frontend for the UI."}
    )


@bp.route("/health")
def health() -> Response:
    """Readiness endpoint consumed by the frontend before enabling processing."""
    try:
        version = metadata.version("brancharchitect")
    except metadata.PackageNotFoundError:
        version = "unknown"

    return jsonify(
        {
            "service": "brancharchitect",
            "status": "ready",
            "ready": True,
            "version": version,
            "capabilities": [
                "tree-stream-upload",
                "sse-progress-stream",
                "msa-tree-inference",
                "tree-interpolation",
            ],
            "routes": {
                "health": "/health",
                "tree_stream": "/treedata/stream",
                "progress_stream": "/stream/progress/<channel_id>",
            },
            "jobs": {
                "msa_tree_inference": _get_msa_tree_inference_status(),
            },
        }
    )


@bp.route("/favicon.ico")
def favicon() -> Response:
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
    tree_inference_engine: str = "iqtree",
    use_gtr: bool = True,
    use_gamma: bool = True,
    iqtree_fast_search: bool = True,
    iqtree_support_mode: str = "none",
    iqtree_ufboot_replicates: int = 1000,
    iqtree_sh_alrt_replicates: int = 1000,
    iqtree_bnni: bool = False,
    use_pseudo: bool = False,
    no_ml: bool = True,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Tuple[Dict[str, Any], list[Dict[str, Any]]]:
    """Run the MSA → tree pipeline and return the stream movie payload.

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
        Tuple of stream metadata and serialized tree list.
    """

    log: Logger = current_app.logger
    temp_dir = tempfile.mkdtemp(prefix="msa-analysis-")

    def report(pct: float, msg: str) -> None:
        if progress_callback:
            progress_callback(pct, msg)

    try:
        report(0, "Starting MSA analysis...")
        log.info("[msa_analysis] Starting MSA analysis...")

        # Output directory for intermediate files (IQ-TREE requires file I/O)
        analysis_output_dir = os.path.join(temp_dir, "output")

        tree_inference_config: FastTreeConfig | IQTreeConfig
        if tree_inference_engine == "fasttree":
            tree_inference_config = FastTreeConfig(
                use_gtr=use_gtr,
                use_gamma=use_gamma,
                use_pseudo=use_pseudo,
                no_ml=no_ml,
            )
        else:
            tree_inference_config = IQTreeConfig(
                use_gtr=use_gtr,
                use_gamma=use_gamma,
                fast_search=iqtree_fast_search,
                support_mode=iqtree_support_mode,
                ufboot_replicates=iqtree_ufboot_replicates,
                sh_alrt_replicates=iqtree_sh_alrt_replicates,
                bnni=iqtree_bnni,
            )

        report(
            10,
            f"Running tree inference pipeline with {tree_inference_config.description} model...",
        )

        log.info(
            "[msa_analysis] Running analysis pipeline with "
            f"{tree_inference_config.description} model..."
        )

        def report_pipeline_progress(pct: float, msg: str) -> None:
            # Keep the MSA tree-inference stage inside _run_msa_analysis_and_interpolate's
            # 10-40 range. Interpolation and streaming own the later ranges.
            report(10 + (pct / 100.0) * 30, msg)

        # Pass MSA content directly - no need to write input file to disk
        pipeline_result = run_pipeline(
            input_file=None,
            output_directory=analysis_output_dir,
            window_size=window_size,
            step_size=window_step,
            fasttree_config=tree_inference_config,
            msa_content=msa_content,  # In-memory content for webservice
            stage_progress_callback=report_pipeline_progress,
        )

        tree_file_path = pipeline_result.tree_file_path

        report(40, "Tree inference complete.")

        # Log dropped taxa information
        if pipeline_result.has_dropped_taxa:
            log.warning(
                f"[msa_analysis] {len(pipeline_result.dropped_taxa)} taxa were dropped "
                f"due to invalid data in some windows: {pipeline_result.dropped_taxa[:5]}..."
            )
            report(
                42,
                f"Note: {len(pipeline_result.dropped_taxa)} taxa dropped (gaps/ambiguous in some windows)",
            )

        if not os.path.exists(tree_file_path):
            raise FileNotFoundError(
                "Analysis script finished but did not produce the expected tree file."
            )

        report(45, "Processing generated trees...")

        with open(tree_file_path, "r", encoding="utf-8") as f:
            tree_content = f.read()

        log.info(
            f"[msa_analysis] Generated {pipeline_result.num_windows} trees with "
            f"{pipeline_result.kept_taxa} taxa. Running interpolation service."
        )

        # Create a sub-callback that maps tree interpolation progress into 45-95.
        tree_progress_callback = _create_sub_progress_callback(
            progress_callback, 45, 95
        )

        metadata, trees = handle_tree_content_streaming(
            tree_content,
            filename=os.path.basename(str(tree_file_path)),
            msa_content=msa_content,
            enable_rooting=enable_rooting,
            window_size=window_size,
            window_step=window_step,
            iqtree_support_mode=(
                iqtree_support_mode if tree_inference_engine == "iqtree" else None
            ),
            progress_callback=tree_progress_callback,
        )

        return metadata, trees
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        log.info(f"[msa_analysis] Cleaned up temporary directory: {temp_dir}")


# ----------------------------------------------------------------------
# Streaming tree processing endpoint
# ----------------------------------------------------------------------


@bp.route("/treedata/stream", methods=["POST"])
def treedata_stream() -> Union[Response, Tuple[dict[str, Any], int]]:
    """
    Start tree processing and return a channel_id immediately.

    Progress updates are sent via SSE on /stream/progress/<channel_id>.
    Metadata and tree chunks are sent before the final complete event.
    """
    log: Logger = current_app.logger
    log.info("[treedata/stream] POST /treedata/stream from %s", request.remote_addr)

    try:
        req_data = parse_tree_data_request(request)
        channel = channels.create()
        claimed_msa_slot = False

        if _is_msa_tree_inference_request(req_data):
            claimed_msa_slot = _claim_msa_tree_inference_slot(channel.channel_id)
            if not claimed_msa_slot:
                channels.remove(channel.channel_id)
                return (
                    _fail(
                        409,
                        "Another MSA tree-inference job is already running. "
                        "Wait for it to finish or restart the BranchArchitect backend before starting a new MSA analysis.",
                    ),
                    409,
                )

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
                        metadata, trees = _run_msa_analysis_and_interpolate(
                            msa_content=req_data.msa_content,
                            window_size=req_data.window_size,
                            window_step=req_data.window_step,
                            enable_rooting=req_data.enable_rooting,
                            tree_inference_engine=req_data.tree_inference_engine,
                            use_gtr=req_data.use_gtr,
                            use_gamma=req_data.use_gamma,
                            iqtree_fast_search=req_data.iqtree_fast_search,
                            iqtree_support_mode=req_data.iqtree_support_mode,
                            iqtree_ufboot_replicates=req_data.iqtree_ufboot_replicates,
                            iqtree_sh_alrt_replicates=req_data.iqtree_sh_alrt_replicates,
                            iqtree_bnni=req_data.iqtree_bnni,
                            use_pseudo=req_data.use_pseudo,
                            no_ml=req_data.no_ml,
                            progress_callback=_make_progress_callback(channel, 10, 85),
                        )

                        send_movie_stream(channel, metadata, trees, log)
                    else:
                        # Tree file processing uses the same streamed movie contract.
                        channel.send_progress(10, "Parsing tree file...")
                        metadata, trees = handle_tree_content_streaming(
                            req_data.tree_content,
                            filename=req_data.tree_filename or "uploaded_file",
                            msa_content=req_data.msa_content,
                            enable_rooting=req_data.enable_rooting,
                            window_size=req_data.window_size,
                            window_step=req_data.window_step,
                            annotate_tree_series_support=True,
                            progress_callback=_make_progress_callback(channel, 10, 85),
                        )

                        send_movie_stream(channel, metadata, trees, log)

                except Exception as e:
                    log.error(
                        "[treedata/stream] Processing error: %s", str(e), exc_info=True
                    )
                    channel.complete(error=str(e))
                finally:
                    if claimed_msa_slot:
                        _release_msa_tree_inference_slot(channel.channel_id)

        # Start background processing
        thread = threading.Thread(target=process_in_background, daemon=True)
        try:
            thread.start()
        except Exception:
            if claimed_msa_slot:
                _release_msa_tree_inference_slot(channel.channel_id)
            channels.remove(channel.channel_id)
            raise

        return jsonify({"channel_id": channel.channel_id})

    except ValueError as e:
        log.warning(f"[treedata/stream] Bad request: {e}")
        return _fail(400, str(e)), 400

    except Exception as e:
        log.error("[treedata/stream] Exception: %s", str(e), exc_info=True)
        return _fail(500, str(e)), 500


@bp.route("/information-scope/analyze", methods=["POST"])
def information_scope_analyze() -> Union[Response, Tuple[dict[str, Any], int]]:
    """
    Start a SatuTe analysis of one tree + one alignment and return a channel_id
    immediately; results arrive on /stream/progress/<channel_id>.

    Streamed rather than synchronous because SatuTe dominates the runtime and
    scales with tree size -- ~6 minutes for the 1871-taxon Tree of Life -- which
    no single request should be made to sit through.
    """
    from webapp.services.trees.information_scope import (
        InformationScopeError,
        run_information_scope_analysis,
    )

    log: Logger = current_app.logger
    log.info(
        "[information-scope/analyze] POST from %s", request.remote_addr
    )

    tree_file = request.files.get("treeFile")
    msa_file = request.files.get("msaFile")
    if not tree_file or not tree_file.filename:
        return _fail(400, "Missing required file 'treeFile'."), 400
    if not msa_file or not msa_file.filename:
        return _fail(400, "Missing required file 'msaFile'."), 400

    tree_content = tree_file.read().decode("utf-8", errors="replace")
    alignment_content = msa_file.read().decode("utf-8", errors="replace")
    if not tree_content.strip():
        return _fail(400, "Uploaded file 'treeFile' is empty."), 400
    if not alignment_content.strip():
        return _fail(400, "Uploaded file 'msaFile' is empty."), 400

    alpha_raw = request.form.get("satuteAlpha", "0.05")
    try:
        alpha = float(alpha_raw)
    except ValueError:
        return _fail(400, "satuteAlpha must be a number."), 400
    if not 0 < alpha < 1:
        return _fail(400, "satuteAlpha must be between 0 and 1."), 400

    model = request.form.get("satuteModel") or None
    # Sliding-window saturation is optional: it needs the per-site export and
    # a second aggregation pass, and on large trees the caller may only want
    # the branch-level result.
    windowed = request.form.get("satuteWindowed", "on") != "off"
    filename = tree_file.filename

    channel = channels.create()
    app: Flask = getattr(current_app, "_get_current_object")()

    def process_in_background() -> None:
        with app.app_context():
            try:
                channel.send_progress(0, "Starting SatuTe analysis...")
                response = run_information_scope_analysis(
                    tree_content=tree_content,
                    alignment_content=alignment_content,
                    filename=filename,
                    alpha=alpha,
                    model=model,
                    logger=log,
                    progress_callback=channel.send_progress,
                    windowed=windowed,
                )
                channel.complete(data=response)
            except InformationScopeError as exc:
                log.warning("[information-scope/analyze] Bad request: %s", exc)
                channel.complete(error=str(exc))
            except Exception as exc:
                log.error(
                    "[information-scope/analyze] Exception: %s", exc, exc_info=True
                )
                channel.complete(error=str(exc))

    thread = threading.Thread(target=process_in_background, daemon=True)
    try:
        thread.start()
    except Exception:
        channels.remove(channel.channel_id)
        raise

    return jsonify({"channel_id": channel.channel_id})


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


@bp.errorhandler(Exception)
def global_error(exc: Exception) -> Tuple[dict[str, Any], int]:
    """Convert uncaught exceptions into the existing short JSON error shape."""
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
    Once connected, the stream emits 'progress', 'metadata', 'trees_chunk', and a
    terminal 'complete' event (with an optional `error` field in its JSON body on
    failure). If channel_id is unknown, the initial response is an HTTP 404 with an
    'error' event instead of a 200 stream; browsers treat any non-2xx response to an
    EventSource's initial connection as a fatal connection failure, so this reaches
    client code as `onerror`, not as a named 'error' event listener.

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

    def stream_and_cleanup() -> Generator[str, None, None]:
        try:
            yield from channel.stream()
        finally:
            channels.remove(channel_id)

    return sse_response(stream_and_cleanup())
