"""Matrix display functionality for logs."""

from typing import Any, List, Callable, Optional, Iterable, TYPE_CHECKING

from brancharchitect.logger.base_logger import AlgorithmLogger
from brancharchitect.logger.formatting import (
    format_partition_set,
    beautify_frozenset,
)

if TYPE_CHECKING:
    from brancharchitect.jumping_taxa.lattice.matrices.types import PMatrix


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


class MatrixLogger(AlgorithmLogger):
    """Extension of AlgorithmLogger with matrix display support."""

    def matrix(
        self,
        matrix: "PMatrix",
        format_func: Optional[Callable[[Iterable[Any]], str]] = None,
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
            # Use ASCII-only brackets/dividers for consistent width across terminals
            line = f"[ {' | '.join(row_vals)} ]"

            terminal_output.append(line)

            # Add dividers between rows (except after last row)
            if r_idx < len(matrix) - 1 and len(matrix) > 1:
                content_width = sum(col_widths) + (len(col_widths) - 1) * 3
                divider = f"[ {'-' * content_width} ]"
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
        self, matrix: List[List[Any]], format_func: Callable[[Any], str]
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
            # Use ASCII-only brackets/dividers for consistent width
            line = f"[ {' | '.join(row_vals)} ]"
            html += line + "\n"

            # Add dividers between rows (except after last row)
            if r_idx < len(matrix) - 1 and len(matrix) > 1:
                content_width = sum(col_widths) + (len(col_widths) - 1) * 3
                divider = f"[ {'-' * content_width} ]"
                html += divider + "\n"

        html += "</pre>\n</div>"
        return html

    def _create_table_matrix(
        self, matrix: List[List[Any]], format_func: Callable[[Any], str]
    ) -> str:
        """Create an HTML table representation of the matrix."""
        # Import here to avoid circular imports
        from brancharchitect.logger.table_logger import TableLogger

        # Create headers
        headers = [""] + [f"Column {i}" for i in range(len(matrix[0]))]

        # Format data for table
        table_data = []
        for i, row in enumerate(matrix):
            table_data.append([f"Row {i}"] + [format_func(cell) for cell in row])

        # Generate HTML table
        table_logger = TableLogger(self.name)
        return table_logger.create_html_table(table_data, headers)

    def log_strategy_selection(self, rows: int, cols: int, category: str) -> None:
        """Log which meet product strategy was selected."""
        if self.disabled:
            return

        if category == "VECTOR":
            self.info("Strategy: Vector meet product for 1×2 matrix")
        elif category == "SQUARE":
            self.info(f"Strategy: Square meet product for {rows}×{cols} matrix")
        elif category == "RECTANGULAR":
            self.info(
                f"Strategy: Rectangular row-wise meet product for {rows}×{cols} matrix"
            )

    def log_results_collection(
        self, main_result: object, counter_result: object
    ) -> int:
        """Log collection of non-empty diagonal results."""
        if self.disabled:
            return 0

        self.info("\n  📋 Collecting Non-Empty Results:")

        count = 0
        if main_result:
            self.info(f"    ✓ Main diagonal solution added: {main_result}")
            count += 1
        else:
            self.info("    ✗ Main diagonal empty - not added")

        if counter_result:
            self.info(f"    ✓ Counter diagonal solution added: {counter_result}")
            count += 1
        else:
            self.info("    ✗ Counter diagonal empty - not added")

        return count

    def log_split_analysis(self, rows: int, cols: int) -> None:
        """Log start of matrix splitting analysis."""
        if self.disabled:
            return

        self.info(f"Analyzing {rows}×{cols} matrix for potential splitting")

    def log_degenerate_extraction(self, count: int) -> None:
        """Log extraction of degenerate rows."""
        if self.disabled:
            return

        self.info(f"Extracted {count} degenerate singleton row(s) as 1×2 matrices")

    def log_grouping_analysis(
        self, left_count: int, right_count: int, selected: str
    ) -> None:
        """Log column grouping analysis results."""
        if self.disabled:
            return

        self.info(f"Grouping by left column resulted in {left_count} groups.")
        self.info(f"Grouping by right column resulted in {right_count} groups.")

        if selected == "right":
            self.info("Selected right column grouping as it is more effective.")
        else:
            self.info("Selected left column grouping.")

    def log_no_split(
        self, reason: str = "single group or no grouping achieved"
    ) -> None:
        """Log when no split is performed."""
        if self.disabled:
            return

        self.info(f"No effective split found - {reason}.")

    def log_split_results(self, count: int) -> None:
        """Log split results summary."""
        if self.disabled:
            return

        self.info(f"Created {count} sub-matrices")

    def log_pairing_mode(self, n: int, mode: str) -> None:
        """Log which pairing mode is being used."""
        if self.disabled:
            return

        if mode == "reverse":
            self.info(f"✓ Using standard reverse mapping for {n} pairs")
            self.info("  • Position i pairs with position (n-1-i)")
            self.info(f"  • Mappings: {[(i, n - 1 - i) for i in range(n)]}")
        elif mode == "asymmetric":
            self.info("📝 Asymmetric Case: Cannot use pure reverse mapping")
            self.info("   Reason: Different number of results from each matrix")
            self.info("   Fallback: Using generalized union approach instead")

    def log_union_creation(self, count: int) -> None:
        """Log creation of union solutions."""
        if self.disabled:
            return

        self.info(f"Creating {count} union solutions")
