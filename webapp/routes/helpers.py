"""Request handling helpers."""

from __future__ import annotations
from dataclasses import dataclass
import os
from typing import Optional, cast
from flask import Request
from werkzeug.datastructures import FileStorage

IQTREE_REPLICATE_COUNT_MIN = 100
IQTREE_REPLICATE_COUNT_MAX = 100000


@dataclass
class TreeDataRequest:
    """Encapsulates data from a tree data upload request."""

    tree_content: Optional[str]  # File content, not FileStorage (for thread safety)
    tree_filename: Optional[str]
    window_size: int
    window_step: int
    enable_rooting: bool
    msa_content: Optional[str] = None
    use_gtr: bool = (
        True  # Use GTR model (General Time Reversible - most realistic for viruses)
    )
    use_gamma: bool = (
        True  # Use gamma rate heterogeneity (accounts for rate variation across sites)
    )
    tree_inference_engine: str = "iqtree"
    iqtree_fast_search: bool = True
    iqtree_support_mode: str = "none"
    iqtree_ufboot_replicates: int = 1000
    iqtree_sh_alrt_replicates: int = 1000
    iqtree_bnni: bool = False
    use_pseudo: bool = False  # Use pseudocounts (recommended for gappy alignments)
    no_ml: bool = True  # Always use no-ML to produce fully bifurcating trees


def get_msa_content(msa_file: Optional[FileStorage]) -> Optional[str]:
    """Extracts content from the uploaded MSA file, if present."""
    if not msa_file or not msa_file.filename:
        return None

    msa_file.seek(0, os.SEEK_END)
    file_size = msa_file.tell()
    msa_file.seek(0)

    if file_size > 0:
        content = msa_file.read()
        if not isinstance(content, bytes):
            raise ValueError("Uploaded file 'msaFile' could not be read as bytes.")
        return content.decode("utf-8", errors="replace")
    return None


def _parse_iqtree_replicate_count(raw_value: object, field_name: str) -> int:
    """Parse an IQ-TREE replicate count from form data with validation."""

    if raw_value is None:
        raise ValueError(f"{field_name} is required.")

    if isinstance(raw_value, bool):
        # bool is an int subclass but not a valid repeat count input in this context.
        raise ValueError(f"{field_name} must be an integer.")

    if isinstance(raw_value, int):
        replicate_count = raw_value
    elif isinstance(raw_value, str):
        stripped = raw_value.strip()
        if not stripped:
            raise ValueError(f"{field_name} must be an integer.")
        try:
            replicate_count = int(stripped)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an integer.") from exc
    elif isinstance(raw_value, bytes):
        try:
            replicate_count = int(raw_value)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an integer.") from exc
    elif isinstance(raw_value, float):
        if not raw_value.is_integer():
            raise ValueError(f"{field_name} must be an integer.")
        replicate_count = int(raw_value)
    else:
        raise ValueError(f"{field_name} must be an integer.")

    if not IQTREE_REPLICATE_COUNT_MIN <= replicate_count <= IQTREE_REPLICATE_COUNT_MAX:
        raise ValueError(
            f"{field_name} must be between {IQTREE_REPLICATE_COUNT_MIN} "
            f"and {IQTREE_REPLICATE_COUNT_MAX}."
        )
    return replicate_count


def _parse_window_size(raw_value: object, field_name: str, default: int) -> int:
    """Parse a positive integer from optional form input."""

    if raw_value is None:
        return default

    if isinstance(raw_value, int):
        value = raw_value
    elif isinstance(raw_value, str):
        stripped = raw_value.strip()
        if not stripped:
            return default
        try:
            value = int(stripped)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an integer.") from exc
    elif isinstance(raw_value, bytes):
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an integer.") from exc
    elif isinstance(raw_value, float):
        if not raw_value.is_integer():
            raise ValueError(f"{field_name} must be an integer.")
        value = int(raw_value)
    else:
        raise ValueError(f"{field_name} must be an integer.")

    if value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")

    return value


def parse_tree_data_request(request: Request) -> TreeDataRequest:
    """Parses and validates the incoming request for tree data processing."""
    tree_file = cast(Optional[FileStorage], request.files.get("treeFile"))
    msa_file = cast(Optional[FileStorage], request.files.get("msaFile"))

    # We need at least one of the two files
    if (not tree_file or not tree_file.filename) and (
        not msa_file or not msa_file.filename
    ):
        raise ValueError(
            "Missing required file. Please provide either a 'treeFile' or an 'msaFile'."
        )

    # If tree_file is provided, validate it
    if tree_file and tree_file.filename:
        tree_file.seek(0, os.SEEK_END)
        if tree_file.tell() == 0:
            raise ValueError("Uploaded file 'treeFile' is empty.")
        tree_file.seek(0)
    else:
        tree_file = None  # Ensure tree_file is None if not provided or empty

    window_size = _parse_window_size(request.form.get("windowSize"), "windowSize", 1)
    window_step = _parse_window_size(
        request.form.get("windowStepSize"), "windowStepSize", 1
    )
    enable_rooting = request.form.get("midpointRooting", "") == "on"

    # Model options for tree inference (GTR and gamma are enabled by default)
    # "on" means checkbox is checked, empty or missing means use default (True)
    use_gtr_raw = request.form.get("useGtr", "on")
    use_gamma_raw = request.form.get("useGamma", "on")
    use_pseudo_raw = request.form.get("usePseudo", "")
    tree_inference_engine = request.form.get("treeInferenceEngine", "iqtree")
    if tree_inference_engine not in {"iqtree", "fasttree"}:
        raise ValueError("Invalid tree inference engine.")
    use_gtr = use_gtr_raw == "on"
    use_gamma = use_gamma_raw == "on"
    use_pseudo = use_pseudo_raw == "on"
    iqtree_fast_search = request.form.get("iqtreeFastSearch", "on") == "on"
    iqtree_support_mode = request.form.get("iqtreeSupportMode", "none")
    if iqtree_support_mode not in {"none", "ufboot", "sh_alrt", "sh_alrt_ufboot"}:
        raise ValueError("Invalid IQ-TREE support mode.")
    iqtree_ufboot_replicates = _parse_iqtree_replicate_count(
        request.form.get("iqtreeUfbootReplicates", 1000),
        "iqtreeUfbootReplicates",
    )
    iqtree_sh_alrt_replicates = _parse_iqtree_replicate_count(
        request.form.get("iqtreeShAlrtReplicates", 1000),
        "iqtreeShAlrtReplicates",
    )
    iqtree_bnni = request.form.get("iqtreeBnni", "") == "on"
    no_ml = request.form.get("noMl", "on") == "on"

    msa_content = get_msa_content(msa_file)

    # Read tree file content now (before request context ends)
    tree_content = None
    tree_filename = None
    if tree_file:
        tree_content = tree_file.read().decode("utf-8", errors="replace")
        tree_filename = tree_file.filename

    return TreeDataRequest(
        tree_content=tree_content,
        tree_filename=tree_filename,
        window_size=window_size,
        window_step=window_step,
        enable_rooting=enable_rooting,
        msa_content=msa_content,
        use_gtr=use_gtr,
        use_gamma=use_gamma,
        tree_inference_engine=tree_inference_engine,
        iqtree_fast_search=iqtree_fast_search,
        iqtree_support_mode=iqtree_support_mode,
        iqtree_ufboot_replicates=iqtree_ufboot_replicates,
        iqtree_sh_alrt_replicates=iqtree_sh_alrt_replicates,
        iqtree_bnni=iqtree_bnni,
        use_pseudo=use_pseudo,
        no_ml=no_ml,
    )
