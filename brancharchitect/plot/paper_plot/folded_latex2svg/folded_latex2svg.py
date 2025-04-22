#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modularized LaTeX to SVG conversion utility.

Based on original work by Tino Wagner and subsequent versions.
This version incorporates improvements based on debugging experience:
- More robust Ghostscript library detection.
- Prioritizes using the --libgs command-line flag for dvisvgm.
- Returns detailed error information in a dictionary instead of raising exceptions directly.
- Uses command lists for subprocess calls.
- Requires the 'lxml' library for SVG manipulation (`pip install lxml`).
- Broken down into smaller helper functions for clarity and maintainability.
"""

import os
import sys
import subprocess
import re
import tempfile
import shutil
from lxml import etree  # Requires: pip install lxml
from ctypes.util import find_library
import random
import string
from typing import Dict, Optional, List, Tuple, Any  # For type hinting

__version__ = "0.6.0 (modularized)"  # Indicate revised version

# --- Defaults (Consider moving to a config class/file later) ---

default_template_revised = r"""
\documentclass[{{ fontsize }}pt,preview]{standalone}
{{ preamble }}
\begin{document}
\begin{preview}
{{ code }}
\end{preview}
\end{document}
"""

default_preamble_revised = r"""
\usepackage[utf8]{inputenc}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{amstext}
\usepackage{newtxtext}
\usepackage[libertine]{newtxmath}
% prevent errors from old font commands
\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
% prevent errors from undefined shortcuts
\newcommand{\N}{\mathbb{N}}
\newcommand{\R}{\mathbb{R}}
\newcommand{\Z}{\mathbb{Z}}
"""

default_params_revised = {
    "fontsize": 12,
    "template": default_template_revised,
    "preamble": default_preamble_revised,
    "latex_cmd_base": "latex",
    "dvisvgm_cmd_base": "dvisvgm",
    "dvisvgm_opts": ["--no-fonts", "--exact-bbox"],
    "scale": 1.0,
    "scour_cmd_base": "scour",
    "svgo_cmd_base": "svgo",
    "optimizer": "scour",
    "libgs": None,
    "timeout": 30,
}

# --- Helper Functions ---


def _find_executable(cmd_base: str) -> Optional[str]:
    """Finds the full path to an executable."""
    return shutil.which(cmd_base)


def _find_ghostscript_library(provided_path: Optional[str] = None) -> Optional[str]:
    """
    Attempts to find a valid path to the Ghostscript shared library.
    Priority: Provided Path > LIBGS Env Var > Standard Paths > find_library
    """
    paths_to_check = []
    if provided_path:
        paths_to_check.append(provided_path)
    libgs_env = os.environ.get("LIBGS")
    if libgs_env:
        paths_to_check.append(libgs_env)

    if sys.platform == "darwin":
        paths_to_check.extend(
            [
                "/opt/homebrew/lib/libgs.dylib",
                "/usr/local/lib/libgs.dylib",
                "/opt/local/lib/libgs.dylib",
            ]
        )
    elif sys.platform.startswith("linux"):
        paths_to_check.extend(
            [
                "/usr/lib/libgs.so",
                "/usr/local/lib/libgs.so",
                "/usr/lib64/libgs.so",
                "/usr/local/lib64/libgs.so",
            ]
        )
    # Add Windows logic here if needed

    try:
        libgs_ctypes = find_library("gs")
        if libgs_ctypes and os.path.isabs(libgs_ctypes):
            paths_to_check.append(libgs_ctypes)
    except Exception:
        pass

    for path in paths_to_check:
        if path and os.path.exists(path):
            return path
    return None


def _merge_params(user_params: Optional[Dict] = None) -> Dict:
    """Merges user parameters with defaults."""
    effective_params = default_params_revised.copy()
    if user_params:
        effective_params.update(user_params)
    return effective_params


def _create_tex_file(
    content: str, directory: str
) -> Tuple[Optional[str], Optional[str]]:
    """Creates the .tex file in the specified directory."""
    tex_filename = os.path.join(directory, "code.tex")
    try:
        with open(tex_filename, "w", encoding="utf-8") as f:
            f.write(content)
        return tex_filename, None  # Return path on success, None error
    except IOError as e:
        return None, f"Failed to write tex file {tex_filename}: {e}"


def _run_subprocess(command: List[str], cwd: str, timeout: Optional[int]) -> Dict:
    """Runs a subprocess, capturing output and handling errors."""
    result_dict = {"stdout": None, "stderr": None, "returncode": 0, "error": None}
    try:
        process_result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        result_dict["stdout"] = process_result.stdout
        result_dict["stderr"] = process_result.stderr
        result_dict["returncode"] = process_result.returncode
        if process_result.returncode != 0:
            # Construct a generic error message, specific command name is in caller
            result_dict["error"] = (
                f"Command failed with exit code {process_result.returncode}"
            )
    except subprocess.TimeoutExpired:
        result_dict["error"] = (
            f"Command '{command[0]}' timed out after {timeout} seconds."
        )
        result_dict["returncode"] = -1  # Indicate timeout
    except FileNotFoundError:
        result_dict["error"] = f"Command not found: {command[0]}"
        result_dict["returncode"] = -1
    except Exception as e:
        result_dict["error"] = (
            f"Failed to run command '{command[0]}': {type(e).__name__}: {e}"
        )
        result_dict["returncode"] = -1
    return result_dict


def _parse_dvisvgm_output(
    stderr_str: str, fontsize: float
) -> Tuple[Optional[float], Optional[float], float]:
    """Parses dvisvgm stderr output for dimensions."""
    width, height, depth = None, None, 0.0
    scaling = 1.00375  # TeX pt -> DTP pt scaling factor
    try:
        size_match = re.search(
            r"graphic size:\s*([0-9.e-]+)pt\s*x\s*([0-9.e-]+)pt", stderr_str or ""
        )
        d_match = re.search(r"\bdepth=([0-9.e-]+)pt", stderr_str or "")
        if size_match:
            width = float(size_match.group(1)) / fontsize * scaling
            height = float(size_match.group(2)) / fontsize * scaling
        if d_match:
            depth = float(d_match.group(1)) / fontsize * scaling
        if width is None or height is None:
            print(
                "Warning: Could not parse width/height from dvisvgm output.",
                file=sys.stderr,
            )
    except Exception as parse_e:
        print(f"Warning: Error parsing dvisvgm output: {parse_e}", file=sys.stderr)
    return width, height, depth


def _modify_svg(
    svg_file_path: str, width: Optional[float], height: Optional[float], depth: float
) -> Optional[str]:
    """Modifies SVG attributes using lxml and returns SVG content as string."""
    try:
        parser = etree.XMLParser(remove_comments=True, recover=True)
        xml = etree.parse(svg_file_path, parser)
        svg_root = xml.getroot()
        for attr in ["width", "height", "style"]:
            if attr in svg_root.attrib:
                del svg_root.attrib[attr]
        if width is not None:
            svg_root.set("width", f"{width:.6f}em")
        if height is not None:
            svg_root.set("height", f"{height:.6f}em")
        if abs(depth) > 1e-6:
            svg_root.set("style", f"vertical-align:{-depth:.6f}em")
        # Return modified SVG as string
        return etree.tostring(
            xml, encoding="unicode", xml_declaration=False, pretty_print=False
        )
    except ImportError:
        print(
            "Warning: lxml not installed (`pip install lxml`). Cannot modify SVG attributes.",
            file=sys.stderr,
        )
        # Fallback: read original file content
        try:
            with open(svg_file_path, "r", encoding="utf-8") as f:
                return f.read()
        except IOError:
            return None  # Failed to read even original
    except Exception as lxml_e:
        print(
            f"Warning: Error processing SVG with lxml: {lxml_e}. Returning original SVG content.",
            file=sys.stderr,
        )
        # Fallback: read original file content
        try:
            with open(svg_file_path, "r", encoding="utf-8") as f:
                return f.read()
        except IOError:
            return None  # Failed to read even original


def _run_optimizer(
    optimizer_name: str,
    cmd_base: str,
    input_svg_path: str,
    output_svg_path: str,
    cwd: str,
    timeout: Optional[int],
) -> Tuple[Optional[str], Optional[str]]:
    """Runs an SVG optimizer (scour or svgo). Returns path to output file or None on error, plus error message."""
    optimizer_exe = _find_executable(cmd_base)
    if not optimizer_exe:
        err_msg = f"Optimizer executable '{cmd_base}' not found. Skipping optimization."
        print(f"Warning: {err_msg}", file=sys.stderr)
        return None, err_msg  # Return None path, error message

    print(f"Running optimizer: {optimizer_name}...")
    cmd_list = []
    if optimizer_name == "scour":
        prefix = "".join(random.choice(string.ascii_letters) for _ in range(3))
        cmd_list = [
            optimizer_exe,
            "--shorten-ids",
            f"--shorten-ids-prefix={prefix}_",
            "--no-line-breaks",
            "--remove-metadata",
            "--enable-comment-stripping",
            "--strip-xml-prolog",
            "-i",
            input_svg_path,
            "-o",
            output_svg_path,
        ]
    elif optimizer_name == "svgo":
        # svgo might need a config file setup - more complex
        # cmd_list = [optimizer_exe, '-i', input_svg_path, '-o', output_svg_path] # Basic example
        err_msg = "svgo optimization not fully implemented in this example."
        print(f"Warning: {err_msg}", file=sys.stderr)
        return None, err_msg  # Not implemented
    else:
        return None, f"Unsupported optimizer: {optimizer_name}"

    result = _run_subprocess(cmd_list, cwd=cwd, timeout=timeout)
    if result["error"]:
        err_msg = f"Optimizer '{optimizer_name}' failed: {result['error']}. Using unoptimized SVG."
        print(f"Warning: {err_msg}", file=sys.stderr)
        # Return None path, include stderr from optimizer if available
        return None, result["stderr"] or err_msg
    else:
        return output_svg_path, None  # Return optimized path, no error


def _read_svg_file(svg_file_path: str) -> Tuple[Optional[str], Optional[str]]:
    """Reads SVG content from a file."""
    try:
        with open(svg_file_path, "r", encoding="utf-8") as f:
            return f.read(), None
    except IOError as e:
        return None, f"Failed to read final SVG file {svg_file_path}: {e}"


# --- Main Modularized Function ---


def latex2svg_modular(
    code: str, params: Optional[Dict] = None, working_directory: Optional[str] = None
) -> Dict:
    """
    Modularized function to convert LaTeX to SVG using latex and dvisvgm.

    Parameters and Returns are the same as latex2svg_revised docstring.
    """
    output_dict = {
        "svg": None,
        "width": None,
        "height": None,
        "valign": None,
        "error": None,
        "log": None,
        "stderr": None,
        "returncode": 0,
    }

    # 1. Handle working directory
    cleanup_needed = False
    actual_working_directory = working_directory
    if actual_working_directory is None:
        try:
            actual_working_directory = tempfile.mkdtemp()
            cleanup_needed = True
        except Exception as e:
            output_dict["error"] = f"Failed to create temporary directory: {e}"
            output_dict["returncode"] = -1
            return output_dict
    elif not os.path.isdir(actual_working_directory):
        output_dict["error"] = (
            f"Provided working directory is not valid: {actual_working_directory}"
        )
        output_dict["returncode"] = -1
        return output_dict

    try:
        # 2. Merge parameters
        effective_params = _merge_params(params)
        fontsize = effective_params["fontsize"]
        timeout = effective_params.get("timeout")

        # 3. Find executables (fallback to common variants)
        latex_exe = _find_executable(effective_params["latex_cmd_base"])
        if not latex_exe:
            latex_exe = _find_executable("pdflatex")
        dvisvgm_exe = _find_executable(effective_params["dvisvgm_cmd_base"])
        if not dvisvgm_exe:
            dvisvgm_exe = _find_executable("dvisvgm")
        if not latex_exe or not dvisvgm_exe:
            output_dict["error"] = (
                f"Required command(s) not found: latex={latex_exe}, dvisvgm={dvisvgm_exe}"
            )
            output_dict["returncode"] = -1
            return output_dict

        # 4. Find Ghostscript library
        libgs_path = _find_ghostscript_library(effective_params.get("libgs"))
        if not libgs_path:
            print("Warning: libgs not found. Conversion might fail.", file=sys.stderr)

        # 5. Construct LaTeX document content
        if "\\documentclass" in effective_params["preamble"]:
            output_dict["error"] = "Do not include \\documentclass in 'preamble'."
            output_dict["returncode"] = -1
            return output_dict
        document = (
            effective_params["template"]
            .replace("{{ preamble }}", effective_params["preamble"])
            .replace("{{ fontsize }}", str(fontsize))
            .replace("{{ code }}", code)
        )

        # Define file paths
        tex_filename = os.path.join(actual_working_directory, "code.tex")
        dvi_filename = os.path.join(actual_working_directory, "code.dvi")
        svg_filename = os.path.join(actual_working_directory, "code.svg")
        opt_svg_filename = os.path.join(actual_working_directory, "optimized.svg")

        # 6. Create .tex file
        _, err = _create_tex_file(document, actual_working_directory)
        if err:
            output_dict["error"] = err
            output_dict["returncode"] = -1
            return output_dict

        # 7. Run LaTeX
        latex_cmd_list = [
            latex_exe,
            "-interaction",
            "nonstopmode",
            "-halt-on-error",
            "code.tex",
        ]
        latex_result = _run_subprocess(
            latex_cmd_list, actual_working_directory, timeout
        )
        output_dict["log"] = latex_result["stdout"]  # Store log
        if latex_result["error"]:
            output_dict.update(latex_result)
            return output_dict  # Update dict with error details
        if not os.path.exists(dvi_filename):
            output_dict["error"] = f"LaTeX ran but DVI file not found: {dvi_filename}"
            output_dict["returncode"] = -1
            return output_dict

        # 8. Run dvisvgm
        dvisvgm_cmd_list = [dvisvgm_exe] + effective_params["dvisvgm_opts"]
        if libgs_path:
            dvisvgm_cmd_list.append("--libgs=" + libgs_path)
        dvisvgm_cmd_list.append("--scale=%f" % effective_params["scale"])
        dvisvgm_cmd_list.extend(["-o", svg_filename, dvi_filename])
        dvisvgm_result = _run_subprocess(
            dvisvgm_cmd_list, actual_working_directory, timeout
        )
        output_dict["stderr"] = dvisvgm_result[
            "stderr"
        ]  # Store stderr (info/warnings/errors)
        if dvisvgm_result["error"]:
            output_dict.update(dvisvgm_result)
            return output_dict
        if not os.path.exists(svg_filename):
            output_dict["error"] = f"dvisvgm ran but SVG file not found: {svg_filename}"
            output_dict["returncode"] = -1
            return output_dict

        # 9. Parse dvisvgm output
        width, height, depth = _parse_dvisvgm_output(
            dvisvgm_result["stderr"] or "", fontsize
        )
        output_dict["width"] = round(width, 6) if width is not None else None
        output_dict["height"] = round(height, 6) if height is not None else None
        output_dict["valign"] = round(-depth, 6)

        # 10. Modify SVG (Optional but recommended for consistent output)
        modified_svg_content = _modify_svg(svg_filename, width, height, depth)
        if modified_svg_content is None:
            # Failed to read/modify, error logged by helper
            # Decide if this is fatal? For now, let's say no, but no SVG content.
            output_dict["error"] = (
                output_dict.get("error") or "Failed to modify/read SVG file"
            )
            # Keep parsed dimensions if available
        else:
            # Write modified content back for optimizer or final result
            try:
                with open(svg_filename, "w", encoding="utf-8") as f:
                    f.write(modified_svg_content)
            except IOError as e:
                output_dict["error"] = (
                    f"Failed to write modified SVG file {svg_filename}: {e}"
                )
                output_dict["returncode"] = -1
                return output_dict

        # 11. Run Optimizer (Optional)
        optimizer = effective_params.get("optimizer", "none").lower()
        final_svg_path = svg_filename  # Default to unoptimized path

        if optimizer in ["scour", "svgo"]:
            opt_path, opt_err = _run_optimizer(
                optimizer,
                effective_params.get(f"{optimizer}_cmd_base", optimizer),
                svg_filename,
                opt_svg_filename,
                actual_working_directory,
                timeout,
            )
            if opt_err:
                # Store optimizer error/warning in stderr field?
                output_dict["stderr"] = (
                    output_dict["stderr"] or ""
                ) + f"\nOptimizer Warning/Error: {opt_err}"
            if opt_path and os.path.exists(opt_path):
                final_svg_path = opt_path  # Use optimized path if successful

        # 12. Read final SVG content
        svg_content, read_err = _read_svg_file(final_svg_path)
        if read_err:
            output_dict["error"] = read_err
            output_dict["returncode"] = -1
            return output_dict
        output_dict["svg"] = svg_content

        # If we reached here without returning, it's a success
        output_dict["returncode"] = 0

    except Exception as outer_e:  # Catch unexpected errors
        output_dict["error"] = f"Unexpected error in latex2svg_modular: {outer_e}"
        output_dict["returncode"] = -1
        import traceback

        output_dict["stderr"] = traceback.format_exc()
    finally:
        # Cleanup
        if (
            cleanup_needed
            and actual_working_directory
            and os.path.exists(actual_working_directory)
        ):
            try:
                shutil.rmtree(actual_working_directory)
            except Exception as cleanup_e:
                print(
                    f"Warning: Failed to clean up temp dir {actual_working_directory}: {cleanup_e}",
                    file=sys.stderr,
                )

    return output_dict


# --- Example Usage ---
if __name__ == "__main__":
    print("Running example usage of latex2svg_modular...")

    # Example 1: Basic usage
    latex_code = r"\frac{a}{b}"
    print(f"\nConverting: {latex_code}")
    result1 = latex2svg_modular(latex_code)
    if result1["error"]:
        print(f"Error: {result1['error']} (Code: {result1['returncode']})")
        if result1["log"]:
            print(f"Log:\n{result1['log']}")
        if result1["stderr"]:
            print(f"Stderr:\n{result1['stderr']}")
    else:
        print(
            f"Success! Width={result1['width']}em, Height={result1['height']}em, VAlign={result1['valign']}em"
        )
        # print("SVG Output:\n", result1['svg'][:200] + "...")

    # Example 2: More complex with custom preamble
    latex_code_complex = r"\oint_C \vec{B} \cdot d\vec{l} = \mu_0 I_{enc}"
    custom_preamble = r"""
                        \usepackage{amsmath}
                        \usepackage{amssymb}
                        \usepackage{esvect} % For vector arrows
                        \usepackage{newtxtext, newtxmath}
                        """
    custom_params = {"preamble": custom_preamble}
    print(f"\nConverting: {latex_code_complex}")
    result2 = latex2svg_modular(latex_code_complex, params=custom_params)
    if result2["error"]:
        print(f"Error: {result2['error']} (Code: {result2['returncode']})")
        if result2["log"]:
            print(f"Log:\n{result2['log']}")
        if result2["stderr"]:
            print(f"Stderr:\n{result2['stderr']}")
    else:
        print(
            f"Success! Width={result2['width']}em, Height={result2['height']}em, VAlign={result2['valign']}em"
        )
        output_filename_example = "example_complex_modular.svg"
        with open(output_filename_example, "w", encoding="utf-8") as f:
            f.write(result2["svg"])
        print(f"Saved complex example SVG to: {output_filename_example}")
