"""Tree visualization and comparison functionality for logs."""

from typing import Any, List, Optional

from brancharchitect.logger.base_logger import AlgorithmLogger
from brancharchitect.tree import Node

# Optional plotting support - commented out since we removed plotting dependencies
# from brancharchitect.plot.tree_plot import plot_rectangular_tree_pair

from brancharchitect.logger.html_content import (
    COMPARE_TREE_SPLIT_CSS,
    TABLE_SPLIT_JS,
    NEWICK_COMBINED_CSS,
    NEWICK_TEMPLATE_COMBINED,
    NEWICK_TEMPLATE_SINGLE_COMBINED,
    NEWICK_HIGHLIGHT_JS,
)


class TreeLogger(AlgorithmLogger):
    """Extension of AlgorithmLogger with tree visualization support."""

    def log_tree_comparison(
        self,
        node_one: Node,
        node_two: Node,
        title: str = "Tree Comparison",
        show_internal_names: bool = False,
        vertical_taxon_labels: bool = False,
    ):
        """Log visual comparison of two trees - rectangular layout only."""
        self.subsection(title)
        
        # Plotting functionality disabled - dependencies removed
        self.add_html("<p><em>Tree plotting disabled - plotting dependencies not available</em></p>")
        return
        
        # # Original plotting code - commented out since we removed plotting dependencies
        # svg_content = plot_rectangular_tree_pair(
        #     node_one, node_two, vertical_leaf_labels=vertical_taxon_labels
        # )
        # # Prefer PNG embedding in HTML; fallback to inline SVG on failure
        # success = self.add_png_from_svg(svg_content)
        # if not success:
        #     self.add_svg(svg_content)

    def compare_tree_splits(
        self,
        tree1: Node,
        tree2: Node,
        sort_by: str = "taxa",
        show_indices: bool = False,
    ):
        """
        Generate an interactive, beautiful comparison table of splits between two trees.

        Args:
            tree1: First tree to compare
            tree2: Second tree to compare
            sort_by: How to sort the splits ("indices", "taxa", "common_first", or "diff_first")
        """
        if self.disabled:
            return

        self.subsection("Tree Split Comparison")

        # Extract splits from both trees
        splits1 = tree1.to_splits()
        splits2 = tree2.to_splits()

        # Get all indices and taxa from both trees
        left_indices = []
        left_taxa = []
        for split in splits1:
            split_indices = frozenset(split.indices)
            left_indices.append(split_indices)
            taxa = frozenset(str(t) for t in split.taxa)
            left_taxa.append(taxa)

        right_indices = []
        right_taxa = []
        for split in splits2:
            split_indices = frozenset(split.indices)
            right_indices.append(split_indices)
            taxa = frozenset(str(t) for t in split.taxa)
            right_taxa.append(taxa)

        # Create composite data for sorting
        from typing import Dict, Any

        all_data: List[Dict[str, Any]] = []

        # First, add all splits from left tree
        for lidx, ltaxa in zip(left_indices, left_taxa):
            # Check if this split exists in the right tree
            if lidx in right_indices:
                r_idx = right_indices.index(lidx)
                all_data.append(
                    {
                        "left_indices": lidx,
                        "left_taxa": ltaxa,
                        "right_indices": right_indices[r_idx],
                        "right_taxa": right_taxa[r_idx],
                        "common": True,
                        "size": int(len(lidx)),
                    }
                )
            else:
                # This split only exists in left tree
                all_data.append(
                    {
                        "left_indices": lidx,
                        "left_taxa": ltaxa,
                        "right_indices": set(),
                        "right_taxa": set(),
                        "common": False,
                        "size": int(len(lidx)),
                    }
                )

        # Now add any splits from right tree that aren't already covered
        for ridx, rtaxa in zip(right_indices, right_taxa):
            if ridx not in left_indices:
                all_data.append(
                    {
                        "left_indices": set(),
                        "left_taxa": set(),
                        "right_indices": ridx,
                        "right_taxa": rtaxa,
                        "common": False,
                        "size": int(len(ridx)),
                    }
                )

        # Sort the data according to the specified criterion
        if sort_by == "indices":
            all_data.sort(
                key=lambda x: (
                    x["size"],
                    tuple(
                        sorted(str(i) for i in x["left_indices"])
                    )  # always returns a tuple of strings
                    if isinstance(x["left_indices"], (set, frozenset))
                    else tuple(),
                )
            )
        elif sort_by == "taxa":
            all_data.sort(
                key=lambda x: (
                    x["size"],
                    tuple(
                        sorted(list(x["left_taxa"]))
                        if isinstance(x["left_taxa"], (set, frozenset))
                        else ()
                    ),
                )
            )
        elif sort_by == "common_first":
            all_data.sort(key=lambda x: (not x["common"], int(x["size"])))
        elif sort_by == "diff_first":
            all_data.sort(key=lambda x: (bool(x["common"]), int(x["size"])))

        # Import here to avoid circular imports
        from brancharchitect.logger.formatting import format_set

        # Generate the HTML table
        # Determine initial active sort button
        sort_indices_active = "active" if sort_by == "indices" else ""
        sort_taxa_active = "active" if sort_by == "taxa" else ""
        sort_common_active = "active" if sort_by == "common_first" else ""
        sort_diff_active = "active" if sort_by == "diff_first" else ""

        # Determine initial column visibility for indices
        idx_active_class = "active" if show_indices else ""
        idx_hidden_class = "" if show_indices else " col-hidden"

        html = f"""
        <div class="split-comparison">
            <div class="controls">
                <div class="sort-options">
                    <span>Sort by:</span>
                    <button class="sort-btn {sort_indices_active}" data-sort="indices">Indices</button>
                    <button class="sort-btn {sort_taxa_active}" data-sort="taxa">Taxa</button>
                    <button class="sort-btn {sort_common_active}" data-sort="common">Common First</button>
                    <button class="sort-btn {sort_diff_active}" data-sort="diff">Differences First</button>
                </div>
                <div class="column-toggle">
                    <span>Toggle columns:</span>
                    <button class="col-toggle-btn {idx_active_class}" data-col="0">Left Indices</button>
                    <button class="col-toggle-btn active" data-col="1">Left Taxa</button>
                    <button class="col-toggle-btn {idx_active_class}" data-col="2">Right Indices</button>
                    <button class="col-toggle-btn active" data-col="3">Right Taxa</button>
                    <button class="col-toggle-btn active" data-col="4">Match</button>
                </div>
            </div>
            <div class="table-wrapper">
                <table class="comparison-table">
                    <thead>
                        <tr>
                            <th class="col-0{idx_hidden_class}">Left Indices</th>
                            <th class="col-1">Left Taxa</th>
                            <th class="col-2{idx_hidden_class}">Right Indices</th>
                            <th class="col-3">Right Taxa</th>
                            <th class="col-4">Match</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        # Add data rows
        for item in all_data:
            left_indices_str = format_set(item["left_indices"]) if show_indices else ""
            left_taxa_str = format_set(set(item["left_taxa"]))
            right_indices_str = (
                format_set(item["right_indices"]) if show_indices else ""
            )
            right_taxa_str = format_set(set(item["right_taxa"]))

            # Determine row class based on match status
            row_class = "common" if item["common"] else "different"
            match_symbol = "✓" if item["common"] else "✗"
            match_class = "match" if item["common"] else "mismatch"

            html += f"""
            <tr class="{row_class}">
                <td class="col-0{idx_hidden_class}">{left_indices_str}</td>
                <td class="col-1">{left_taxa_str}</td>
                <td class="col-2{idx_hidden_class}">{right_indices_str}</td>
                <td class="col-3">{right_taxa_str}</td>
                <td class="col-4 {match_class}">{match_symbol}</td>
            </tr>
            """

        # Close table
        html += """
                    </tbody>
                </table>
            </div>
        </div>
        """

        # Calculate summary statistics
        common_splits = sum(1 for item in all_data if item["common"])
        total_splits = len(all_data)

        summary = f"""
        <div class="summary-box">
            <p><strong>Summary:</strong> {common_splits} of {total_splits} splits are common ({(common_splits / total_splits * 100):.1f}%)</p>
            <p><strong>Left Tree:</strong> {len(splits1)} splits</p>
            <p><strong>Right Tree:</strong> {len(splits2)} splits</p>
            <p><strong>Unique to Left:</strong> {len(splits1) - common_splits}</p>
            <p><strong>Unique to Right:</strong> {len(splits2) - common_splits}</p>
        </div>
        """

        # Add CSS via logger and inject the HTML and JS separately
        self.add_css(COMPARE_TREE_SPLIT_CSS)
        self.raw_html(html + summary)
        self.raw_html(TABLE_SPLIT_JS)

    def log_lattice_edge_tables(
        self,
        edge: Any,
        *,
        show_common_covers: bool = True,
        show_unique_min_covers: bool = True,
        show_atoms: bool = False,
        tablefmt: str = "html",
    ) -> None:
        """Log tables for a lattice edge: common covers, unique minimum covers, and atoms.

        Options:
            show_common_covers: Include tables for left/right common covers.
            show_unique_min_covers: Include tables for minimum covers of unique splits per side.
            show_atoms: Include atom tables (minimal elements) of the unique split sets.
            tablefmt: Table format for terminal/HTML output (default 'html').
        """
        if self.disabled:
            return

        from brancharchitect.logger.formatting import format_set as _fmt

        self.subsection("Lattice Edge Tables")
        # Edge header
        try:
            split_taxa = getattr(edge, "pivot_split", getattr(edge, "split", None))
            split_taxa = getattr(split_taxa, "taxa", set())
            self.info(f"Pivot Split: {_fmt(split_taxa)}")
        except Exception:
            self.info(f"Pivot Split: {getattr(edge, 'split', 'N/A')}")

        # Helper: compute unique splits and their minimum cover per side
        def _min_cover_unique(node_a: Node, node_b: Node):
            try:
                a_s = node_a.to_splits()
                b_s = node_b.to_splits()
                uniq = a_s - b_s
                return uniq.minimum_cover(), uniq
            except Exception:
                return None, None

        # Build a single combined table
        left_covers = []
        right_covers = []
        if show_common_covers:
            try:
                left_covers = [
                    _fmt(set(cov))
                    for cov in (getattr(edge, "t1_common_covers", []) or [])
                ]
                right_covers = [
                    _fmt(set(cov))
                    for cov in (getattr(edge, "t2_common_covers", []) or [])
                ]
            except Exception:
                left_covers, right_covers = [], []

        left_min = []
        right_min = []
        left_atoms = []
        right_atoms = []

        if show_unique_min_covers or show_atoms:
            # Handle both old attribute names (t1_node, t2_node) and new names (tree1_node, tree2_node)
            tree1_node = getattr(edge, "tree1_node", getattr(edge, "t1_node", None))
            tree2_node = getattr(edge, "tree2_node", getattr(edge, "t2_node", None))

            if tree1_node is not None and tree2_node is not None:
                t1_min, t1_uniq = _min_cover_unique(tree1_node, tree2_node)
            else:
                t1_min, t1_uniq = None, None
            if tree2_node is not None and tree1_node is not None:
                t2_min, t2_uniq = _min_cover_unique(tree2_node, tree1_node)
            else:
                t2_min, t2_uniq = None, None

            if show_unique_min_covers:
                if t1_min:
                    left_min = [_fmt(set(p.taxa)) for p in t1_min]
                if t2_min:
                    right_min = [_fmt(set(p.taxa)) for p in t2_min]

            if show_atoms:
                if t1_uniq:
                    try:
                        left_atoms = [
                            _fmt(set(p.taxa)) for p in t1_uniq.minimal_elements()
                        ]
                    except Exception:
                        left_atoms = []
                if t2_uniq:
                    try:
                        right_atoms = [
                            _fmt(set(p.taxa)) for p in t2_uniq.minimal_elements()
                        ]
                    except Exception:
                        right_atoms = []

        # Normalize to a single table with aligned rows
        def _get(lst: List[str], i: int) -> str:
            return lst[i] if i < len(lst) else ""

        n_rows = max(
            len(left_covers),
            len(right_covers),
            len(left_min),
            len(right_min),
            len(left_atoms),
            len(right_atoms),
            1,
        )

        combined_rows: List[List[str]] = []
        # First column: Pivot split (only for first row to avoid repetition)
        _pivot_obj = getattr(edge, "pivot_split", None)
        if _pivot_obj is None:
            _pivot_obj = getattr(edge, "split", None)
        split_str = (
            _fmt(getattr(_pivot_obj, "taxa", set())) if _pivot_obj is not None else ""
        )
        for i in range(n_rows):
            combined_rows.append(
                [
                    split_str if i == 0 else "",
                    _get(left_covers, i),
                    _get(right_covers, i),
                    _get(left_min, i),
                    _get(right_min, i),
                    _get(left_atoms, i),
                    _get(right_atoms, i),
                ]
            )

        headers = [
            "Pivot Split",
            "L Common Cover",
            "R Common Cover",
            "L Unique Min",
            "R Unique Min",
            "L Atoms",
            "R Atoms",
        ]

        self.table(combined_rows, headers=headers, tablefmt=tablefmt)

    def log_newick_strings(
        self, tree1: Node, tree2: Optional[Node] = None, title: str = "Newick Strings"
    ):
        """
        Log newick string representation of trees with copy functionality.

        Args:
            tree1: First tree
            tree2: Optional second tree for comparison
            title: Title for the section
        """
        if self.disabled:
            return

        self.subsection(title)

        # Get Newick string representations properly
        # The to_newick method already adds a semicolon at the end
        newick1 = tree1.to_newick(lengths=False)

        # Enhance the CSS with explicit white text color and better formatting
        enhanced_css = NEWICK_COMBINED_CSS.replace(
            ".combined-content {", ".combined-content { color: white; font-size: 14px; "
        )

        # Ensure proper spacing and fix line breaks
        if tree2:
            newick2 = tree2.to_newick(lengths=False)
            # Create HTML with properly formatted trees
            # Inject CSS separately and then add HTML + JS
            self.add_css(enhanced_css)
            html_content = NEWICK_TEMPLATE_COMBINED.format(
                newick1.strip(), newick2.strip()
            )
            self.raw_html(html_content)
            self.raw_html(NEWICK_HIGHLIGHT_JS)
        else:
            # Single tree case
            self.add_css(enhanced_css)
            html_content = NEWICK_TEMPLATE_SINGLE_COMBINED.format(newick1.strip())
            self.raw_html(html_content)
            self.raw_html(NEWICK_HIGHLIGHT_JS)

    def table(
        self,
        rows: List[List[str]],
        headers: Optional[List[str]] = None,
        tablefmt: str = "simple",
    ) -> None:
        """Log a table in the specified format."""
        if tablefmt == "html":
            # Generate HTML table
            html = "<table>\n"
            if headers:
                html += "<thead><tr>\n"
                for header in headers:
                    html += f"<th>{header}</th>\n"
                html += "</tr></thead>\n"
            html += "<tbody>\n"
            for row in rows:
                html += "<tr>\n"
                for cell in row:
                    html += f"<td>{cell}</td>\n"
                html += "</tr>\n"
            html += "</tbody>\n</table>\n"
            self.raw_html(html)
        else:
            # Use tabulate for other formats
            try:
                import tabulate

                if headers is not None:
                    table_str = tabulate.tabulate(
                        rows, headers=headers, tablefmt=tablefmt
                    )
                else:
                    table_str = tabulate.tabulate(rows, tablefmt=tablefmt)
                self.info(table_str)
            except ImportError:
                # Fallback to simple text table
                if headers:
                    self.info(" | ".join(headers))
                    self.info("-" * len(" | ".join(headers)))
                for row in rows:
                    self.info(" | ".join(str(cell) for cell in row))
