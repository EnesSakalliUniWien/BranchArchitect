"""Request handling helpers."""

from __future__ import annotations
from dataclasses import dataclass
import os
from typing import Optional, cast
from flask import Request
from werkzeug.datastructures import FileStorage


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
        return msa_file.read().decode("utf-8", errors="replace")
    return None


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

    window_size = int(request.form.get("windowSize", 1))
    window_step = int(request.form.get("windowStepSize", 1))
    enable_rooting = request.form.get("midpointRooting", "") == "on"

    # Model options for tree inference (GTR and gamma are enabled by default)
    # "on" means checkbox is checked, empty or missing means use default (True)
    use_gtr_raw = request.form.get("useGtr", "on")
    use_gamma_raw = request.form.get("useGamma", "on")
    use_pseudo_raw = request.form.get("usePseudo", "")
    use_gtr = use_gtr_raw == "on"
    use_gamma = use_gamma_raw == "on"
    use_pseudo = use_pseudo_raw == "on"
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
        use_pseudo=use_pseudo,
        no_ml=no_ml,
    )
