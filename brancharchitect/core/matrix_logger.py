"""Matrix display functionality for logs."""

import re
from typing import Any, List, Callable, Optional

from brancharchitect.core.base_logger import AlgorithmLogger, beautify_frozenset
from brancharchitect.jumping_taxa.lattice.types import PMatrix


def to_latex_matrix(
    matrix: List[List[Any]], format_func: Optional[Callable[[Any], str]] = None
) -> str:
    """Convert a matrix to LaTeX representation."""
    if format_func is None:
        format_func = beautify_frozenset

    rows = []
    for row in matrix:
        formatted_row = [format_func(cell) for cell in row]
        # Replace special characters for LaTeX compatibility
        latex_row = [
            cell.replace("{", "\\{")
            .replace("}", "\\}")
            .replace("(", "\\left(")
            .replace(")", "\\right)")
            .replace("∅", "\\emptyset")
            for cell in formatted_row
        ]
        rows.append(" & ".join(latex_row))

    latex_code = "\\begin{bmatrix}\n"
    latex_code += " \\\\\n".join(rows)
    latex_code += "\n\\end{bmatrix}"

    return latex_code


def format_partition(part):
    """Format a Partition (or its tuple representation) as '(a, b, ...)'."""
    # Check if it's a Partition object with reverse_encoding
    if hasattr(part, 'reverse_encoding') and hasattr(part, 'indices'):
        try:
            # Use the reverse_encoding to get taxon names
            reverse_encoding = part.reverse_encoding
            taxa_names = sorted(reverse_encoding.get(i, str(i)) for i in part.indices)
            return "(" + ", ".join(taxa_names) + ")"
        except Exception:
            # Fall back to indices if something goes wrong
            pass
    
    # Default behavior for non-Partition objects or if reverse_encoding fails
    try:
        values = tuple(part)  # works if part is iterable (like Partition)
    except TypeError:
        values = part
    return "(" + ", ".join(str(x) for x in values) + ")"


def format_partition_set(ps):
    """Format a PartitionSet as a brace-enclosed, comma-separated list of partitions."""
    parts = sorted(ps, key=lambda p: format_partition(p))
    return "{" + ", ".join(format_partition(p) for p in parts) + "}"


def format_matrix(matrix: PMatrix) -> str:
    """
    Given a 2x2 matrix (list of lists) of PartitionSet objects,
    return a string representation that looks like:

      ⎡ cell00         │ cell01 ⎤
      ⎢─────────────────────────────⎥
      ⎣ cell10         │ cell11 ⎦
    """
    formatted = [[format_partition_set(cell) for cell in row] for row in matrix]
    col_widths = [max(len(row[j]) for row in formatted) for j in range(2)]

    def pad(cell, width):
        return cell.ljust(width)

    top = (
        "⎡ "
        + pad(formatted[0][0], col_widths[0])
        + "   │ "
        + pad(formatted[0][1], col_widths[1])
        + " ⎤"
    )
    bottom = (
        "⎣ "
        + pad(formatted[1][0], col_widths[0])
        + "   │ "
        + pad(formatted[1][1], col_widths[1])
        + " ⎦"
    )
    sep_width = len(top) - 2
    middle = "⎢ " + "─" * (sep_width - 4) + " ⎥"
    return "\n".join([top, middle, bottom])


class MatrixLogger(AlgorithmLogger):
    """Extension of AlgorithmLogger with matrix display support."""

    from typing import Callable, Optional

    def matrix(
        self,
        matrix: PMatrix,
        format_func: Optional[Callable[[object], str]] = None,
        title: str = "",
    ) -> None:
        """
        Display a matrix with ASCII art in terminal and mathematical notation in HTML.
        For HTML, provides optional toggle to view ASCII, table, or LaTeX representations.
        """
        if self.disabled or not matrix:
            return

        # Default to the beautify_frozenset function if not provided
        if format_func is None:
            format_func = format_partition_set

        # For terminal output - create ASCII representation
        if title:
            self.logger.info(f"\n{title}:")

        # Calculate column widths for terminal display
        col_widths = []
        for col in range(len(matrix[0])):
            col_width = max(
                len(format_func(matrix[row][col])) for row in range(len(matrix))
            )
            col_widths.append(max(col_width + 4, 12))  # Add padding

        # Generate ASCII art for terminal
        terminal_output = []
        for r_idx, row in enumerate(matrix):
            # Format row values with proper spacing
            row_vals = [
                format_func(cell).ljust(width) for cell, width in zip(row, col_widths)
            ]

            # Add appropriate brackets based on position
            if r_idx == 0 and len(matrix) > 1:
                line = f"⎡ {' │ '.join(row_vals)} ⎤"
            elif r_idx == len(matrix) - 1 and len(matrix) > 1:
                line = f"⎣ {' │ '.join(row_vals)} ⎦"
            elif len(matrix) > 1:
                line = f"⎢ {' │ '.join(row_vals)} ⎥"
            else:
                line = f"[ {' │ '.join(row_vals)} ]"

            terminal_output.append(line)

            # Add dividers between rows (except after last row)
            if r_idx < len(matrix) - 1 and len(matrix) > 1:
                divider = (
                    "⎢ " + "─" * (sum(col_widths) + (len(col_widths) - 1) * 3) + " ⎥"
                )
                terminal_output.append(divider)

        # Print ASCII art to terminal
        self.logger.info("\n".join(terminal_output))

        # Generate LaTeX representation for HTML
        latex_matrix = to_latex_matrix(matrix, format_func)

        # For HTML output - create a container with the mathematical view visible
        matrix_wrapper = '<div class="matrix-container">'

        if title:
            matrix_wrapper += f"<h4>{title}</h4>"

        # Add toggle controls - Added "LaTeX" button
        matrix_wrapper += """
        <div class="matrix-toggle">
            <div class="toggle-buttons">
                <button class="toggle-button active" data-view="mathjax">Mathematical</button>
                <button class="toggle-button" data-view="ascii">ASCII</button>
                <button class="toggle-button" data-view="table">Table</button>
                <button class="toggle-button" data-view="latex">LaTeX</button>
            </div>
        </div>
        """

        # Primary mathematical view using MathJax
        matrix_wrapper += (
            f'<div class="mathjax-view matrix-view">$$\n{latex_matrix}\n$$</div>'
        )

        # Create ASCII view (hidden initially)
        ascii_view = self._create_ascii_matrix(matrix, format_func)
        matrix_wrapper += f'<div class="ascii-view matrix-view" style="display:none;">{ascii_view}</div>'

        # Create table view (hidden initially)
        table_view = self._create_table_matrix(matrix, format_func)
        matrix_wrapper += f'<div class="table-view matrix-view" style="display:none;">{table_view}</div>'

        # Create LaTeX code view (hidden initially)
        matrix_wrapper += f'<div class="latex-view matrix-view" style="display:none;"><pre>{latex_matrix}</pre></div>'

        matrix_wrapper += "</div>"

        self._html_content.append(matrix_wrapper)

    def _create_ascii_matrix(
        self, matrix: List[List[Any]], format_func: Callable
    ) -> str:
        """Create an ASCII art representation of the matrix."""
        # Calculate column widths
        col_widths = []
        for col in range(len(matrix[0])):
            col_width = max(
                len(format_func(matrix[row][col])) for row in range(len(matrix))
            )
            col_widths.append(max(col_width + 4, 12))  # Add padding

        # Generate ASCII with proper borders and alignment
        html = '<div class="ascii-matrix">\n<pre>'

        for r_idx, row in enumerate(matrix):
            # Format row values with proper spacing
            row_vals = [
                format_func(cell).ljust(width) for cell, width in zip(row, col_widths)
            ]

            # Add appropriate brackets based on position
            if r_idx == 0 and len(matrix) > 1:
                line = f"⎡ {' │ '.join(row_vals)} ⎤"
            elif r_idx == len(matrix) - 1 and len(matrix) > 1:
                line = f"⎣ {' │ '.join(row_vals)} ⎦"
            elif len(matrix) > 1:
                line = f"⎢ {' │ '.join(row_vals)} ⎥"
            else:
                line = f"[ {' │ '.join(row_vals)} ]"

            # Add HTML styling
            formatted_line = re.sub(
                r"([⎡⎢⎣⎤⎥⎦\[\]])", r'<span class="bracket">\1</span>', line
            )
            formatted_line = re.sub(
                r"(│)", r'<span class="divider">│</span>', formatted_line
            )
            formatted_line = re.sub(
                r"(\{[^{}]*\})", r'<span class="set">\1</span>', formatted_line
            )
            formatted_line = re.sub(
                r"(\([^()]+\))", r'<span class="element">\1</span>', formatted_line
            )

            html += formatted_line + "\n"

            # Add dividers between rows (except after last row)
            if r_idx < len(matrix) - 1 and len(matrix) > 1:
                divider = (
                    "⎢ " + "─" * (sum(col_widths) + (len(col_widths) - 1) * 3) + " ⎥"
                )
                divider = re.sub(
                    r"([⎡⎢⎣⎤⎥⎦\[\]])", r'<span class="bracket">\1</span>', divider
                )
                divider = re.sub(r"(─)", r'<span class="divider">─</span>', divider)
                html += divider + "\n"

        html += "</pre>\n</div>"
        return html

    def _create_table_matrix(
        self, matrix: List[List[Any]], format_func: Callable
    ) -> str:
        """Create an HTML table representation of the matrix."""
        # Import here to avoid circular imports
        from brancharchitect.core.table_logger import TableLogger

        # Create headers
        headers = [""] + [f"Column {i}" for i in range(len(matrix[0]))]

        # Format data for table
        table_data = []
        for i, row in enumerate(matrix):
            table_data.append([f"Row {i}"] + [format_func(cell) for cell in row])

        # Generate HTML table
        table_logger = TableLogger(self.name)
        return table_logger._create_html_table(table_data, headers)

    def matrix_pretty(self, matrix: List[List[Any]], title: str = ""):
        """
        Display a matrix of PartitionSet objects using pretty printing.
        This uses the custom format_matrix function.
        """
        if self.disabled or not matrix:
            return

        if title:
            self.info(f"\n{title}:")

        # Generate the formatted matrix string
        pretty_output = format_matrix(matrix)

        # Log it to the terminal
        self.logger.info(pretty_output)

        # Include it in the HTML output as a preformatted code block
        self._html_content.append(f'<pre class="matrix">{pretty_output}</pre>')
