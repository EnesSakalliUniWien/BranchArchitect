"""Core logging functionality."""

import logging
import re
from tabulate import tabulate
from typing import Any, List, Set, Callable
from functools import wraps
from brancharchitect.tree import Node
from brancharchitect.plot.tree_plot import plot_rectangular_tree_pair


def format_set(s: Set[Any]) -> str:
    """Format set for consistent display."""
    if not s:
        return "∅"
    return "{" + ", ".join(str(x) for x in sorted(s)) + "}"


def beautify_frozenset(obj: Any) -> str:
    """Convert frozenset objects to beautiful mathematical notation."""
    if obj is None:
        return "∅"

    # Convert to string first
    s = str(obj)

    # Replace frozenset({...}) with just {...}
    s = re.sub(r"frozenset\(\{(.+?)\}\)", r"{\1}", s)

    # Replace nested frozensets recursively
    while "frozenset" in s:
        s = re.sub(r"frozenset\(\{(.+?)\}\)", r"{\1}", s)

    # Replace double parentheses around single elements
    s = re.sub(r"\(\((.+?)\)\)", r"(\1)", s)

    # Clean up any remaining artifacts
    s = s.replace("()", "∅")
    s = s.replace("{}", "∅")

    return s


def to_latex_matrix(matrix: List[List[Any]], format_func: Callable = None) -> str:
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


class AlgorithmLogger:
    """Base logger class for algorithm visualization and debugging."""

    def __init__(self, name: str):
        self.name = name
        self.disabled = False
        self._html_content = ['<div class="content">']
        self._css_content = []

        # Include MathJax for mathematical notation
        self._include_mathjax = True
        self._mathjax_header = """
        <script>
        MathJax = {
          tex: {
            inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
            displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
            processEscapes: true,
            processEnvironments: true,
            packages: ['base', 'ams', 'noerrors', 'noundefined']
          },
          svg: {
            fontCache: 'global',
            scale: 1.1
          },
          startup: {
            typeset: true
          }
        };
        </script>
        <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
        """

        # Create logger with single handler to avoid duplication
        self.logger = logging.getLogger(name)
        self.logger.handlers = []  # Clear any existing handlers

        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        # Prevent propagation to avoid duplicate logs
        self.logger.propagate = False

        # Add enhanced CSS for matrix display
        self._css_content.append(
            """
            /* Base styles */
            .split-analysis {
                margin-top: 1em;
                padding: 1em;
                background: #2d2d2d;
                border-radius: 4px;
            }
            
            .error-detail {
                margin: 1em 0;
                padding: 1em;
                background: #331f1f;
                border-radius: 4px;
            }
            
            .error-detail p {
                margin: 0.5em 0;
            }
            
            .error-detail strong {
                color: #ff8080;
            }
            
            /* MathJax matrix container - primary display */
            .mathjax-matrix {
                margin: 2em 0;
                padding: 20px;
                background: #1e293b;
                border-radius: 8px;
                border: 1px solid #3a4a6d;
                text-align: center;
                overflow-x: auto;
                box-shadow: 0 4px 12px rgba(0,0,0,0.25);
            }
            
            .mathjax-matrix .MJX-TEX {
                font-size: 120% !important;
            }
            
            /* Matrix view toggle controls */
            .matrix-toggle {
                margin: 1.5em 0 1em;
                text-align: right;
            }
            
            .toggle-buttons {
                display: inline-flex;
                background: #2c3445;
                border-radius: 6px;
                padding: 4px;
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
            }
            
            .toggle-button {
                background: transparent;
                border: none;
                color: #8a9bbc;
                padding: 8px 16px;
                border-radius: 4px;
                cursor: pointer;
                font-family: 'Menlo', 'DejaVu Sans Mono', monospace;
                font-size: 0.9em;
                transition: all 0.2s ease;
            }
            
            .toggle-button:hover {
                color: #dbe4fd;
                background: #3a4559;
            }
            
            .toggle-button.active {
                background: #394963;
                color: #ffffff;
                box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);
            }
            
            /* Alternative views - hidden by default */
            .ascii-matrix, .matrix-table {
                display: none;
            }
            
            /* ASCII Matrix (shown when toggled) */
            .ascii-matrix {
                margin: 1em 0;
                padding: 25px 30px;
                background: #282c34;
                border-radius: 8px;
                border-left: 4px solid #61afef;
            }
            
            .ascii-matrix pre {
                font-family: 'DejaVu Sans Mono', 'Fira Code', 'Courier New', monospace;
                font-size: 1.1em;
                line-height: 1.5;
                margin: 0;
                color: #e0e0e0;
                white-space: pre;
            }
            
            /* Bracket coloring */
            .bracket {
                color: #e5c07b;
                font-weight: bold;
                font-size: 1.4em;
            }
            
            .set {
                color: #98c379;
            }
            
            .element {
                color: #61afef;
            }
            
            .matrix-divider {
                color: #636b7c;
            }
        """
        )

        # Add enhanced CSS for matrix display
        self._css_content.append(
            """
            /* Core styles */
            .matrix-container {
                margin: 2em 0;
                position: relative;
            }
            
            /* Matrix toggle controls */
            .matrix-toggle {
                margin-bottom: 1em;
                text-align: right;
            }
            
            .toggle-buttons {
                display: inline-flex;
                background: #2c3445;
                border-radius: 6px;
                padding: 4px;
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
            }
            
            .toggle-button {
                background: transparent;
                border: none;
                color: #8a9bbc;
                padding: 8px 12px;
                border-radius: 4px;
                cursor: pointer;
                transition: all 0.2s ease;
            }
            
            .toggle-button:hover {
                color: #dbe4fd;
                background: #3a4559;
            }
            
            .toggle-button.active {
                background: #394963;
                color: #ffffff;
                box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);
            }
            
            /* Matrix views */
            .mathjax-view {
                padding: 20px 30px;
                background: #1c2333;
                background: linear-gradient(145deg, #1a1e2d 0%, #252b3b 100%);
                border-radius: 8px;
                border: 1px solid #3a4a6d;
                text-align: center;
                overflow-x: auto;
                box-shadow: 0 4px 12px rgba(0,0,0,0.25);
            }
            
            .mathjax-view .MJX-TEX {
                font-size: 120% !important;
            }
            
            .ascii-view pre {
                background: #282c34;
                padding: 15px;
                border-radius: 5px;
                font-family: 'DejaVu Sans Mono', monospace;
                overflow-x: auto;
                white-space: pre;
            }
            
            .bracket {
                color: #e5c07b;
                font-weight: bold;
            }
            
            .set {
                color: #98c379;
            }
            
            .divider {
                color: #636b7c;
            }
            
            .element {
                color: #61afef;
            }
        """
        )

    def section(self, title: str):
        """Create a new section in the log."""
        if self.disabled:
            return
        self.logger.info(f"\n{'='*20} {title} {'='*20}\n")
        self._html_content.append(f'<section class="section"><h3>{title}</h3>')

    def info(self, message: str):
        """Log info message."""
        if self.disabled:
            return
        self.logger.info(message)
        self._html_content.append(f'<p class="info">{message}</p>')

    def warning(self, message: str):
        """Log warning message."""
        if self.disabled:
            return
        self.logger.warning(message)
        self._html_content.append(f'<p class="warning">{message}</p>')

    def table(
        self,
        data: List[List[Any]],
        headers: List[str] = None,
        title: str = None,
        tablefmt: str = "grid",
        colalign=None,
    ):
        """Display data as a formatted table."""
        if self.disabled:
            return

        if title:
            self.logger.info(f"\n{title}:")

        if tablefmt == "html":
            html_table = self._create_html_table(data, headers)
            self.raw_html(html_table)
        else:
            ascii_table = tabulate(
                data, headers=headers, tablefmt=tablefmt, colalign=colalign
            )
            self.info(ascii_table)

        if title:
            self._html_content.append(f"<h4>{title}</h4>")

    def add_svg(self, svg_content: str):
        """Add SVG visualization to the debug output."""
        if self.disabled:
            return
        self._html_content.append(f'<div class="svg-container">{svg_content}</div>')

    def result(self, label: str, value: Any):
        """Log a result with a label."""
        if self.disabled:
            return
        self.logger.info(f"{label}: {value}")
        self._html_content.append(
            f'<div class="result"><strong>{label}:</strong> {value}</div>'
        )

    def raw_html(self, html_content: str):
        """Add raw HTML content to the debug output."""
        if self.disabled:
            return
        self._html_content.append(html_content)

    def add_css(self, css: str):
        """Add custom CSS to the debug output."""
        if self.disabled:
            return
        self._css_content.append(css)

    def get_html_content(self) -> str:
        """Get the accumulated HTML content with MathJax if enabled."""
        # Close all open sections and the content div
        self._html_content.append("</section>")
        self._html_content.append("</div>")

        # Add toggle script at the end - this is the only place we need it
        toggle_script = """
        <script>
        document.addEventListener('DOMContentLoaded', function() {
            // Set up matrix toggle buttons
            const toggleButtons = document.querySelectorAll('.matrix-toggle .toggle-button');
            toggleButtons.forEach(button => {
                button.addEventListener('click', function() {
                    // Find the parent matrix container
                    const matrixContainer = this.closest('.matrix-container');
                    if (!matrixContainer) return;
                    
                    // Get the view to show
                    const viewType = this.getAttribute('data-view');
                    
                    // Update active button state
                    const buttons = matrixContainer.querySelectorAll('.toggle-button');
                    buttons.forEach(btn => btn.classList.remove('active'));
                    this.classList.add('active');
                    
                    // Hide all views in this container
                    const views = matrixContainer.querySelectorAll('.matrix-view');
                    views.forEach(view => view.style.display = 'none');
                    
                    // Show the selected view
                    const selectedView = matrixContainer.querySelector('.' + viewType + '-view');
                    if (selectedView) {
                        selectedView.style.display = 'block';
                        
                        // If it's MathJax, trigger typesetting
                        if (viewType === 'mathjax' && typeof MathJax !== 'undefined') {
                            MathJax.typeset([selectedView]);
                        }
                    }
                });
            });
        });
        </script>
        """

        # If MathJax is enabled, include the header at the beginning
        if self._include_mathjax:
            return (
                self._mathjax_header
                + "\n"
                + "\n".join(self._html_content)
                + toggle_script
            )
        else:
            return "\n".join(self._html_content) + toggle_script

    def get_css_content(self) -> str:
        """Get the accumulated CSS content."""
        return "\n".join(self._css_content)

    def clear(self):
        """Clear all accumulated content."""
        self._html_content = []
        self._css_content = []

    def _create_html_table(self, data: List[List[Any]], headers: List[str]) -> str:
        """Create HTML table string with dynamic column widths based on content."""

        # Calculate maximum content width for each column
        col_widths = []
        num_cols = len(headers) if headers else len(data[0])

        for col in range(num_cols):
            # Get max width from header
            max_width = len(str(headers[col])) if headers else 0

            # Compare with content widths
            for row in data:
                content_width = len(str(row[col]))
                max_width = max(max_width, content_width)

            # Add some padding
            col_widths.append(max_width + 20)  # 20px padding

        # Create table with container
        html_parts = [
            """
            <div class="table-container" style="
                width: auto !important;
                max-width: 100% !important;
                overflow-x: auto !important;
                margin: 1em 0;
            ">
            """
        ]

        # Add table with nowrap and fixed layout
        html_parts.append(
            """
            <table class="table table-striped table-bordered" style="
                table-layout: fixed !important;
                white-space: nowrap !important;
                width: auto !important;
                min-width: fit-content !important;
                border-collapse: collapse !important;
            ">
        """
        )

        # Add headers
        if headers:
            html_parts.append("<thead>")
            html_parts.append("<tr>")
            for header, width in zip(headers, col_widths):
                html_parts.append(
                    f"""
                    <th style="
                        min-width: {width}px !important;
                        padding: 8px 15px !important;
                        white-space: nowrap !important;
                        overflow: visible !important;
                    ">{header}</th>
                """
                )
            html_parts.append("</tr>")
            html_parts.append("</thead>")

        # Add body
        html_parts.append("<tbody>")
        for row in data:
            html_parts.append("<tr>")
            for cell, width in zip(row, col_widths):
                html_parts.append(
                    f"""
                    <td style="
                        min-width: {width}px !important;
                        padding: 8px 15px !important;
                        white-space: nowrap !important;
                        overflow: visible !important;
                    ">{cell}</td>
                """
                )
            html_parts.append("</tr>")
        html_parts.append("</tbody>")

        html_parts.append("</table>")
        html_parts.append("</div>")

        return "\n".join(html_parts)

    def comparison_analysis(self, results: list, inter: list, sym: list):
        """Log comparison results analysis with proper table formatting."""
        if self.disabled:
            return

        # Generate comparison table
        self.table(
            results,
            headers=["Covet", "Intersection", "A-B", "B-A"],
            title="Comparison Results",
            tablefmt="grid",
            colalign=("left", "left", "left", "left"),
        )

    def log_execution(self, func):
        """Decorator for logging function execution."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            self.section(f"Executing {func.__name__}")
            try:
                result = func(*args, **kwargs)
                self.info(f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                self.info(f"Error in {func.__name__}: {str(e)}")
                raise

        return wrapper

    def code(self, code_str: str, language: str = "python", title: str = None) -> None:
        """Display code with syntax highlighting."""
        if self.disabled:
            return

        if title:
            self.info(f"\n{title}:")

        self.logger.info(f"\n{code_str}")

        # Add formatted code to HTML output
        if title:
            self._html_content.append(f"<h4>{title}</h4>")

        self._html_content.append(
            f"""<pre class="code-block {language}"><code>{code_str}</code></pre>"""
        )

    def print_sedge_comparison(self, s_edge):
        """Print comparison of s-edge trees using rectangular layout."""
        if self.disabled:
            return

        self.section("S-Edge Tree Comparison")

        try:
            # Generate SVG visualization using plot_rectangular_tree_pair
            from brancharchitect.plot.tree_plot import plot_rectangular_tree_pair

            if hasattr(s_edge, "node_one") and hasattr(s_edge, "node_two"):
                svg_content = plot_rectangular_tree_pair(
                    s_edge.node_one,
                    s_edge.node_two,
                    width=800,
                    height=400,
                    margin=30,
                    label_offset=2,
                )
                self.add_svg(svg_content)

                # Add text description
                self.info(f"Edge Type One: {s_edge.edge_type_one}")
                self.info(f"Edge Type Two: {s_edge.edge_type_two}")
                if hasattr(s_edge, "split"):
                    self.info(f"Split: {s_edge.split}")

        except Exception as e:
            self.info(f"Failed to generate s-edge visualization: {str(e)}")
            self.info(str(s_edge))  # Fallback to string representation

    def matrix(self, matrix: List[List[Any]], format_func=None, title: str = None):
        """
        Display a matrix with ASCII art in terminal and mathematical notation in HTML.
        For HTML, provides optional toggle to view ASCII or table representations.
        """
        if self.disabled or not matrix:
            return

        # Default to the beautify_frozenset function if not provided
        if format_func is None:
            format_func = beautify_frozenset

        # For terminal output - create ASCII representation
        if title:
            self.logger.info(f"\n{title}:")
            
        # Calculate column widths for terminal display
        col_widths = []
        for col in range(len(matrix[0])):
            col_width = max(len(format_func(matrix[row][col])) for row in range(len(matrix)))
            col_widths.append(max(col_width + 4, 12))  # Add padding

        # Generate ASCII art for terminal
        terminal_output = []
        for r_idx, row in enumerate(matrix):
            # Format row values with proper spacing
            row_vals = [format_func(cell).ljust(width) for cell, width in zip(row, col_widths)]
            
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
                divider = "⎢ " + "─" * (sum(col_widths) + (len(col_widths) - 1) * 3) + " ⎥"
                terminal_output.append(divider)

        # Print ASCII art to terminal
        self.logger.info("\n".join(terminal_output))

        # Generate LaTeX representation for HTML
        latex_matrix = to_latex_matrix(matrix, format_func)

        # For HTML output - create a container with the mathematical view visible
        matrix_wrapper = '<div class="matrix-container">'

        if title:
            matrix_wrapper += f"<h4>{title}</h4>"

        # Add toggle controls - we need consistent class names
        matrix_wrapper += """
        <div class="matrix-toggle">
            <div class="toggle-buttons">
                <button class="toggle-button active" data-view="mathjax">Mathematical</button>
                <button class="toggle-button" data-view="ascii">ASCII</button>
                <button class="toggle-button" data-view="table">Table</button>
            </div>
        </div>
        """

        # Primary mathematical view using MathJax (use consistent class name)
        matrix_wrapper += (
            f'<div class="mathjax-view matrix-view">$$\n{latex_matrix}\n$$</div>'
        )

        # Create ASCII view (hidden initially)
        ascii_view = self._create_ascii_matrix(matrix, format_func)
        matrix_wrapper += f'<div class="ascii-view matrix-view" style="display:none;">{ascii_view}</div>'

        # Create table view (hidden initially)
        table_view = self._create_table_matrix(matrix, format_func)
        matrix_wrapper += f'<div class="table-view matrix-view" style="display:none;">{table_view}</div>'

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
        # Create headers
        headers = [""] + [f"Column {i}" for i in range(len(matrix[0]))]

        # Format data for table
        table_data = []
        for i, row in enumerate(matrix):
            table_data.append([f"Row {i}"] + [format_func(cell) for cell in row])

        # Generate HTML table
        return self._create_html_table(table_data, headers)

    def log_bidirectional_analysis(self, direction_by_intersection):
        """
        Create a comprehensive HTML table displaying bidirectional analysis data.

        Shows relationships between sets A and B, with detailed information about:
        - Common elements and differences
        - Direction information
        - Taxon indices and names
        - Relationship analysis

        Args:
            direction_by_intersection: List of dictionaries containing bidirectional data
        """
        if not direction_by_intersection:
            self.info("No bidirectional analysis data available.")
            return

        self.section("Bidirectional Analysis")

        # Step 1: Create a comprehensive table
        headers = [
            "Pair",
            "A Elements",
            "B Elements",
            "A ∩ B",
            "A - B",
            "B - A",
        ]

        # Format each row of data with more comprehensive information
        table_rows = []

        for idx, entry in enumerate(direction_by_intersection, 1):

            # Get basic data
            pair = entry.get("pair", "N/A")

            set_a = entry.get("A", set())
            set_b = entry.get("B", set())

            common = set_a.intersection(set_b) if set_a and set_b else set()

            a_minus_b = set_a - set_b if set_a else set()
            b_minus_a = set_b - set_a if set_b else set()

            row = [
                str(pair),
                f"{format_set(set_a)}",
                f"{format_set(set_b)}",
                f"{format_set(common)}",
                f"{format_set(a_minus_b)}",
                f"{format_set(b_minus_a)}",
            ]
            table_rows.append(row)

        # Generate table with fancy formatting
        self.table(
            table_rows,
            headers=headers,
            title="Comprehensive Direction Analysis",
            tablefmt="html",
        )

    def print_trees_side_by_side(self, tree1, tree2, show_internal_names=False):
        # Generate tree representations
        self.log_tree_comparison(tree1, tree2, show_internal_names=show_internal_names)

    def html(self, html_content: str):
        """Add HTML content to the debug output (alias for raw_html)."""
        self.raw_html(html_content)

    def end_section(self):
        """End the current section."""
        if self.disabled:
            return
        self._html_content.append("</div>")

    def error(self, message: str):
        """Log an error message."""
        if self.disabled:
            return
        self.logger.error(message)
        self._html_content.append(f'<p class="error">{message}</p>')

    def log_tree_comparison(
        self, node_one: Node, node_two: Node, title: str = "Tree Comparison"
    ):
        """Log visual comparison of two trees - rectangular layout only."""
        self.section(title)
        svg_content = plot_rectangular_tree_pair(node_one, node_two)
        self.add_svg(svg_content)

    def log_meet_result(self, meet_result, computation_steps=None):
        """Log the meet product result with detailed computation steps."""
        # Show final result
        self.info(f"Final Result: {format_set(meet_result)}")
        # Create results table
        self.table(
            [[format_set(meet_result)]],
            headers=["Meet Product Result"],
            title="Final Computation Result",
            tablefmt="grid",
        )

    def log_covet(self, arms_t_one: List[Any], arms_t_two: List[Any]):
        """Log the arms definition table."""
        arm_headers = [
            [f"A{i}", format_set(arm)] for i, arm in enumerate(arms_t_one, 1)
        ] + [[f"B{i}", format_set(arm)] for i, arm in enumerate(arms_t_two, 1)]
        self.table(arm_headers, headers=["Tree Arms", "Sets"], title="Arms Definition")