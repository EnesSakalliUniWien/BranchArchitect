"""Table display functionality for logs."""

from typing import Any, List, Optional
from tabulate import tabulate
from brancharchitect.core.base_logger import AlgorithmLogger, format_set
import random


class TableLogger(AlgorithmLogger):
    """Extension of AlgorithmLogger with table support."""

    def table(
        self,
        data: List[List[Any]],
        headers: Optional[List[str]] = None,
        title: Optional[str] = None,
        tablefmt: str = "grid",
        colalign=None,
    ):
        """Display data as a formatted table."""
        if self.disabled:
            return

        if headers is None:
            headers = []

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

    def log_combined_data(
        self,
        arms_t_one: Optional[List[Any]] = None,
        arms_t_two: Optional[List[Any]] = None,
        t1_unique_atoms: Optional[List[Any]] = None,
        t2_unique_atoms: Optional[List[Any]] = None,
        t1_unique_covers: Optional[List[Any]] = None,
        t2_unique_covers: Optional[List[Any]] = None,
        name: str = "Combined Analysis Data",
        look_up=None,
    ):
        """
        Log all data tables in a vertically aligned manner.
        
        This combines the functionality of log_covet and log_unique_atoms_and_covets
        to display all tables vertically aligned and sequentially with labels on the left side.

        Args:
            arms_t_one: List of PartitionSets for the left tree's arms
            arms_t_two: List of PartitionSets for the right tree's arms
            t1_unique_atoms: List of PartitionSets unique to the left tree's atoms
            t2_unique_atoms: List of PartitionSets unique to the right tree's atoms
            t1_unique_covers: List of PartitionSets unique to the left tree's covets
            t2_unique_covers: List of PartitionSets unique to the right tree's covets
            name: Name for the display section
            look_up: Optional dictionary for encoding/decoding
        """
        if self.disabled:
            return

        self.section(f"{name} - Visual Representation")

        # Create a single HTML table with all tables vertically stacked
        html_parts = [
            """
            <div class="combined-tables-container" style="
                width: auto !important;
                max-width: 100% !important;
                overflow-x: auto !important;
                margin: 1em 0;
            ">
            <table class="combined-tables" style="
                border-collapse: collapse !important;
                width: auto !important;
                table-layout: fixed !important;
            ">
            """
        ]

        # 1. First table row: Arms Definition (if provided)
        if arms_t_one is not None and arms_t_two is not None:
            left_count = len(arms_t_one)
            right_count = len(arms_t_two)
            
            # Create the header cells
            column_headers = []
            for i in range(1, left_count + 1):
                column_headers.append(f"L{i}")
            for i in range(1, right_count + 1):
                column_headers.append(f"R{i}")
            
            # Create content cells
            content_cells = []
            for arm in arms_t_one:
                content_cells.append(format_set(arm))
            for arm in arms_t_two:
                content_cells.append(format_set(arm))
            
            # Add the row with label
            html_parts.append('<tr style="border-bottom: 2px solid #ccc;">')
            html_parts.append('<td style="padding: 10px; font-weight: bold; text-align: right; white-space: nowrap; vertical-align: middle;">Tree Arms</td>')
            
            # Add column headers
            for header in column_headers:
                html_parts.append(f'<td style="padding: 8px; text-align: center; font-weight: bold; min-width: 80px;">{header}</td>')
            
            html_parts.append('</tr>')
            
            # Add data row
            html_parts.append('<tr style="border-bottom: 20px solid transparent;">')
            html_parts.append('<td style="padding: 10px;"></td>')  # Empty cell for alignment with label
            
            for cell in content_cells:
                html_parts.append(f'<td style="padding: 8px; text-align: center; min-width: 80px;">{cell}</td>')
            
            html_parts.append('</tr>')

        # 2. Second table row: Unique Atoms (if provided)
        if t1_unique_atoms is not None and t2_unique_atoms is not None:
            left_count = len(t1_unique_atoms)
            right_count = len(t2_unique_atoms)
            
            # Create the header cells
            column_headers = []
            for i in range(1, left_count + 1):
                column_headers.append(f"L{i}")
            for i in range(1, right_count + 1):
                column_headers.append(f"R{i}")
            
            # Create content cells
            content_cells = []
            for atom in t1_unique_atoms:
                content_cells.append(format_set(atom))
            for atom in t2_unique_atoms:
                content_cells.append(format_set(atom))
            
            # Add the row with label
            html_parts.append('<tr style="border-bottom: 2px solid #ccc;">')
            html_parts.append('<td style="padding: 10px; font-weight: bold; text-align: right; white-space: nowrap; vertical-align: middle;">Unique Atoms</td>')
            
            # Add column headers
            for header in column_headers:
                html_parts.append(f'<td style="padding: 8px; text-align: center; font-weight: bold; min-width: 80px;">{header}</td>')
            
            html_parts.append('</tr>')
            
            # Add data row
            html_parts.append('<tr style="border-bottom: 20px solid transparent;">')
            html_parts.append('<td style="padding: 10px;"></td>')  # Empty cell for alignment with label
            
            for cell in content_cells:
                html_parts.append(f'<td style="padding: 8px; text-align: center; min-width: 80px;">{cell}</td>')
            
            html_parts.append('</tr>')
        
        # 3. Third table row: Unique Covets (if provided)
        if t1_unique_covers is not None and t2_unique_covers is not None:
            left_count = len(t1_unique_covers)
            right_count = len(t2_unique_covers)
            
            # Create the header cells
            column_headers = []
            for i in range(1, left_count + 1):
                column_headers.append(f"L{i}")
            for i in range(1, right_count + 1):
                column_headers.append(f"R{i}")
            
            # Create content cells
            content_cells = []
            for covet in t1_unique_covers:
                content_cells.append(format_set(covet))
            for covet in t2_unique_covers:
                content_cells.append(format_set(covet))
            
            # Add the row with label
            html_parts.append('<tr style="border-bottom: 2px solid #ccc;">')
            html_parts.append('<td style="padding: 10px; font-weight: bold; text-align: right; white-space: nowrap; vertical-align: middle;">Unique Covets</td>')
            
            # Add column headers
            for header in column_headers:
                html_parts.append(f'<td style="padding: 8px; text-align: center; font-weight: bold; min-width: 80px;">{header}</td>')
            
            html_parts.append('</tr>')
            
            # Add data row
            html_parts.append('<tr style="border-bottom: 20px solid transparent;">')
            html_parts.append('<td style="padding: 10px;"></td>')  # Empty cell for alignment with label
            
            for cell in content_cells:
                html_parts.append(f'<td style="padding: 8px; text-align: center; min-width: 80px;">{cell}</td>')
            
            html_parts.append('</tr>')
        
        # Close the table and container
        html_parts.append('</table>')
        html_parts.append('</div>')
        
        # Add the HTML to the log
        self.raw_html("\n".join(html_parts))

        # Generate copyable code representation if needed
        if look_up:
            self.section(f"{name} - Copyable Code")
            lookup_dict = look_up
            lookup_var_name = f"lookup_{random.randint(1000, 9999)}"

            # Format partitionsets into code representation with correct tuple formatting
            def format_partitionset_as_code(pset):
                # Extract the indices and convert to properly formatted tuples
                tuples_str = []
                for partition in pset:
                    if hasattr(partition, "indices"):
                        indices = tuple(sorted(partition.indices))
                        if len(indices) == 1:
                            # Single element tuple needs trailing comma
                            tuples_str.append(f"({indices[0]},)")
                        else:
                            tuples_str.append(str(indices))
                    else:
                        # Fallback if indices not found
                        tuples_str.append(str(partition))

                return f"PartitionSet({{{', '.join(tuples_str)}}}, {lookup_var_name})"

            # Create the lookup definition with all entries
            lookup_str = f"{lookup_var_name} = {{"
            lookup_items = []

            # Sort keys to make output deterministic
            for k, v in sorted(lookup_dict.items(), key=lambda x: x[1]):
                lookup_items.append(f'"{k}": {v}')

            lookup_str += ", ".join(lookup_items)
            lookup_str += "}"

            # Prepare the code template with conditional sections based on what data was provided
            code_parts = [f"{lookup_str}\n\n{{"]
            code_parts.append(f'    "name": "{name}",')
            
            # Only include sections for data that was provided
            if arms_t_one is not None and arms_t_two is not None:
                code_parts.append('    "left_covets": [')
                code_parts.append(f"        {',        '.join(format_partitionset_as_code(arm) for arm in arms_t_one)}")
                code_parts.append('    ],')
                code_parts.append('    "right_covets": [')
                code_parts.append(f"        {',        '.join(format_partitionset_as_code(arm) for arm in arms_t_two)}")
                code_parts.append('    ],')
                # Include expected results only if arms data is present
                code_parts.append('    "expected": (True, True, True, False),  # Update with expected results')
            
            if t1_unique_atoms is not None and t2_unique_atoms is not None:
                code_parts.append('    "t1_unique_atoms": [')
                code_parts.append(f"        {',        '.join(format_partitionset_as_code(atom) for atom in t1_unique_atoms)}")
                code_parts.append('    ],')
                code_parts.append('    "t2_unique_atoms": [')
                code_parts.append(f"        {',        '.join(format_partitionset_as_code(atom) for atom in t2_unique_atoms)}")
                code_parts.append('    ],')
            
            if t1_unique_covers is not None and t2_unique_covers is not None:
                code_parts.append('    "t1_unique_covers": [')
                code_parts.append(f"        {',        '.join(format_partitionset_as_code(covet) for covet in t1_unique_covers)}")
                code_parts.append('    ],')
                code_parts.append('    "t2_unique_covers": [')
                code_parts.append(f"        {',        '.join(format_partitionset_as_code(covet) for covet in t2_unique_covers)}")
                code_parts.append('    ],')
            
            code_parts.append("},")
            
            code_template = "\n".join(code_parts)

            # Import CSS style from html_content to maintain consistency
            from brancharchitect.core.html_content import UNIQUE_ATOMS_COVETS_CODE_CSS

            self.html(
                f"""
            <div class="code-container">
                <div class="code-header">
                    <span>Copyable test case:</span>
                    <button class="copy-button" onclick="copyCode(this)">Copy</button>
                </div>
                <pre class="code-block"><code>{code_template}</code></pre>
            </div>
            <style>
            {UNIQUE_ATOMS_COVETS_CODE_CSS}
            </style>
            <script>
            function copyCode(button) {{
                const codeBlock = button.parentElement.nextElementSibling.textContent;
                navigator.clipboard.writeText(codeBlock);
                button.textContent = 'Copied!';
                setTimeout(() => {{
                    button.textContent = 'Copy';
                }}, 2000);
            }}
            </script>
            """
            )

    def log_map_details(
        self, intersection_map, left_minus_right_map, right_minus_left_map
    ):
        """
        Log each of the three maps in a detailed manner.
        """
        if self.disabled:
            return

        self.section("Detailed Maps Logging")

        # Log sizes in a quick table
        table_data = [
            ["intersection_map", len(intersection_map)],
            ["left_minus_right_map", len(left_minus_right_map)],
            ["right_minus_left_map", len(right_minus_left_map)],
        ]
        self.table(
            table_data, headers=["Map", "Size"], title="Map Sizes", tablefmt="html"
        )

        # Import here to avoid circular imports
        from brancharchitect.core.formatting import beautify_frozenset, format_set

        # Function to format a map entry
        def format_map_entry(key, val):
            return [
                f"{beautify_frozenset(key)}",
                f"{format_set(val['cover_left'])}",
                f"{format_set(val['cover_right'])}",
                f"{format_set(val['left_only'])}",
                f"{format_set(val['right_only'])}",
                f"{val['index'] if 'index' in val else 'N/A'}",
            ]

        # Helper to log a map in detail
        def log_single_map(map_data, title_str):
            rows = []
            for k, v in map_data.items():
                rows.append(format_map_entry(k, v))
            if rows:
                self.section(title_str)
                self.table(
                    rows,
                    headers=["Key", "cover_left", "cover_right", "left_only", "right_only", "index"],
                    tablefmt="html",
                )

        log_single_map(intersection_map, "intersection_map Details")
        log_single_map(left_minus_right_map, "left_minus_right_map Details")
        log_single_map(right_minus_left_map, "right_minus_right_map Details")

    # Keep these methods for backward compatibility
    def log_covet(
        self,
        arms_t_one: List[Any],
        arms_t_two: List[Any],
        name: str = "Covet example",
        look_up=None,
    ):
        """
        Log the arms definition table with both visualization and copyable code representation.
        
        This method is maintained for backward compatibility and delegates to log_combined_data.
        """
        self.log_combined_data(
            arms_t_one=arms_t_one,
            arms_t_two=arms_t_two,
            name=name,
            look_up=look_up
        )

    def log_unique_atoms_and_covets(
        self,
        t1_unique_atoms: List[Any],
        t2_unique_atoms: List[Any],
        t1_unique_covers: List[Any],
        t2_unique_covers: List[Any],
        name: str = "Unique Atoms and Covets",
        look_up=None,
    ):
        """
        Log the unique atoms and covets for both left and right trees.
        
        This method is maintained for backward compatibility and delegates to log_combined_data.
        """
        self.log_combined_data(
            t1_unique_atoms=t1_unique_atoms,
            t2_unique_atoms=t2_unique_atoms,
            t1_unique_covers=t1_unique_covers,
            t2_unique_covers=t2_unique_covers,
            name=name,
            look_up=look_up
        )
