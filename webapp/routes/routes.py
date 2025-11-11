# --------------------------------------------------------------
#  routes.py
# --------------------------------------------------------------
from __future__ import annotations

from logging import Logger
import json
from pathlib import Path
from typing import Dict, Any
from flask import Response
from flask import Blueprint, current_app, jsonify, request, send_from_directory
from brancharchitect.io import UUIDEncoder
from webapp.routes.helpers import parse_tree_data_request
from webapp.services.tree_processing_service import handle_uploaded_file
from typing import Union, Tuple

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


# ----------------------------------------------------------------------
# Main business endpoint – tree upload & analysis
# ----------------------------------------------------------------------


@bp.route("/treedata", methods=["POST"])
def treedata() -> Union[Response, Tuple[dict[str, Any], int]]:
    log: Logger = current_app.logger
    log.info("[treedata] POST /treedata from %s", request.remote_addr)

    try:
        req_data = parse_tree_data_request(request)

        log.info(
            f"[treedata] Processing file: {req_data.tree_file.filename}, "
            f"Rooting: {req_data.enable_rooting}, "
            f"Window: {req_data.window_size}, Step: {req_data.window_step}, "
            f"MSA provided: {req_data.msa_content is not None}"
        )

        response_data: Dict[str, Any] = handle_uploaded_file(
            req_data.tree_file,
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
