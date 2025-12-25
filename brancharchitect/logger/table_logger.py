"""Table display functionality for logs.

This module renders small tables for terminal and HTML logs. By default, any
explicit depiction of numeric indices is disabled to keep the output focused on
human‑readable labels and partition names.
"""

from typing import Any, List, Optional, Sequence
from tabulate import tabulate
from brancharchitect.logger.base_logger import AlgorithmLogger


class TableLogger(AlgorithmLogger):
    """Extension of AlgorithmLogger with table support.

    Attributes:
        show_indices: When True, logger may render auxiliary index columns or
            numeric index representations in specialized views. Default False.
    """

    # Global toggle to control any explicit index rendering in this logger.
    # Kept as a class attribute so existing call sites don't need to pass
    # constructor arguments.
    show_indices: bool = False

    def table(
        self,
        data: List[List[Any]],
        headers: Optional[List[str]] = None,
        title: Optional[str] = None,
        tablefmt: str = "grid",
        colalign: Optional[Sequence[Optional[str]]] = None,
    ) -> None:
        """Display data as a formatted table."""
        if self.disabled:
            return

        if headers is None:
            headers = []

        if title:
            self.logger.info(f"\n{title}:")

        if tablefmt == "html":
            if title:
                # For HTML tables, render title before the table as a header
                self._html_content.append(f"<h4>{title}</h4>")
            html_table = self._create_html_table(data, headers)
            self.raw_html(html_table)
        else:
            ascii_table = tabulate(
                data,
                headers=headers,
                tablefmt=tablefmt,
                colalign=colalign,
                showindex=False,
            )
            self.info(ascii_table)

        # In non-HTML formats, optionally echo a small header below for readability
        if title and tablefmt != "html":
            self._html_content.append(f"<h4>{title}</h4>")

    def _create_html_table(self, data: List[List[Any]], headers: List[str]) -> str:
        """Create HTML table string with dynamic column widths based on content."""

        # Calculate maximum content width for each column
        col_widths: List[int] = []
        num_cols = len(headers) if headers else (len(data[0]) if data else 0)

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
