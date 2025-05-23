"""Tree visualization and comparison functionality for logs."""

from typing import Any, List, Optional

from brancharchitect.core.base_logger import AlgorithmLogger
from brancharchitect.tree import Node
from brancharchitect.plot.tree_plot import plot_rectangular_tree_pair
from brancharchitect.core.html_content import (
    COMPARE_TREE_SPLIT_CSS,
    TABLE_SPLIT_JS,
    NEWICK_COMBINED_CSS,
    NEWICK_TEMPLATE_COMBINED,
    NEWICK_TEMPLATE_SINGLE_COMBINED,
    NEWICK_HIGHLIGHT_JS,
)


class TreeLogger(AlgorithmLogger):
    """Extension of AlgorithmLogger with tree visualization support."""

    def print_sedge_comparison(self, s_edge):
        """Print comparison of s-edge trees using rectangular layout."""
        if self.disabled:
            return

        self.section("S-Edge Tree Comparison")

        try:
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

    def print_trees_side_by_side(self, tree1, tree2, show_internal_names=False):
        """Print two trees side by side."""
        self.log_tree_comparison(tree1, tree2)  # removed show_internal_names

    def log_tree_comparison(
        self,
        node_one: Node,
        node_two: Node,
        title: str = "Tree Comparison",
        show_internal_names=False,
    ):
        """Log visual comparison of two trees - rectangular layout only."""
        self.section(title)
        svg_content = plot_rectangular_tree_pair(node_one, node_two)
        self.add_svg(svg_content)

    def log_solutions_for_sub_lattice(self, s_edge, solutions):
        """Log solutions for a sub-lattice."""
        if self.disabled:
            return
        self.section(
            f"Solutions found for s_edge: {s_edge.split if hasattr(s_edge, 'split') else s_edge}"
        )
        for sol in solutions:
            self.info(f"Solution: {sol}")

    def log_cover_cartesian_product(
        self, t1_common_covers: List[Any], t2_common_covers: List[Any]
    ):
        """Log the Cartesian product of covers."""
        self.section("Cartesian Product of Covers")
        for i, left_set in enumerate(t1_common_covers):
            for j, right_set in enumerate(t2_common_covers):
                self.info(
                    f"Pair: (Left {i}, Right {j}) => Left: {left_set}, Right: {right_set}"
                )

    def compare_tree_splits(self, tree1: Node, tree2: Node, sort_by: str = "indices"):
        """
        Generate an interactive, beautiful comparison table of splits between two trees.

        Args:
            tree1: First tree to compare
            tree2: Second tree to compare
            sort_by: How to sort the splits ("indices", "taxa", "common_first", or "diff_first")
        """
        if self.disabled:
            return

        self.section("Tree Split Comparison")

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
        all_data = []

        # First, add all splits from left tree
        for i, (lidx, ltaxa) in enumerate(zip(left_indices, left_taxa)):
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
                        "size": len(lidx),
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
                        "size": len(lidx),
                    }
                )

        # Now add any splits from right tree that aren't already covered
        for i, (ridx, rtaxa) in enumerate(zip(right_indices, right_taxa)):
            if ridx not in left_indices:
                all_data.append(
                    {
                        "left_indices": set(),
                        "left_taxa": set(),
                        "right_indices": ridx,
                        "right_taxa": rtaxa,
                        "common": False,
                        "size": len(ridx),
                    }
                )

        # Sort the data according to the specified criterion
        if sort_by == "indices":
            all_data.sort(
                key=lambda x: (
                    x["size"],
                    tuple(
                        sorted(list(x["left_indices"]))
                        if isinstance(x["left_indices"], (set, frozenset))
                        else ()
                    ),
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
            all_data.sort(key=lambda x: (not x["common"], x["size"]))
        elif sort_by == "diff_first":
            all_data.sort(key=lambda x: (x["common"], x["size"]))

        # Import here to avoid circular imports
        from brancharchitect.core.base_logger import format_set

        # Generate the HTML table
        html = """
        <div class="split-comparison">
            <div class="controls">
                <div class="sort-options">
                    <span>Sort by:</span>
                    <button class="sort-btn active" data-sort="indices">Indices</button>
                    <button class="sort-btn" data-sort="taxa">Taxa</button>
                    <button class="sort-btn" data-sort="common">Common First</button>
                    <button class="sort-btn" data-sort="diff">Differences First</button>
                </div>
                <div class="column-toggle">
                    <span>Toggle columns:</span>
                    <button class="col-toggle-btn active" data-col="0">Left Indices</button>
                    <button class="col-toggle-btn active" data-col="1">Left Taxa</button>
                    <button class="col-toggle-btn active" data-col="2">Right Indices</button>
                    <button class="col-toggle-btn active" data-col="3">Right Taxa</button>
                    <button class="col-toggle-btn active" data-col="4">Match</button>
                </div>
            </div>
            <div class="table-wrapper">
                <table class="comparison-table">
                    <thead>
                        <tr>
                            <th class="col-0">Left Indices</th>
                            <th class="col-1">Left Taxa</th>
                            <th class="col-2">Right Indices</th>
                            <th class="col-3">Right Taxa</th>
                            <th class="col-4">Match</th>
                        </tr>
                    </thead>
                    <tbody>
        """

        # Add data rows
        for item in all_data:
            left_indices_str = format_set(item["left_indices"])
            left_taxa_str = format_set(item["left_taxa"])
            right_indices_str = format_set(item["right_indices"])
            right_taxa_str = format_set(item["right_taxa"])

            # Determine row class based on match status
            row_class = "common" if item["common"] else "different"
            match_symbol = "✓" if item["common"] else "✗"
            match_class = "match" if item["common"] else "mismatch"

            html += f"""
            <tr class="{row_class}">
                <td class="col-0">{left_indices_str}</td>
                <td class="col-1">{left_taxa_str}</td>
                <td class="col-2">{right_indices_str}</td>
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

        # Add everything to the page
        self.raw_html(COMPARE_TREE_SPLIT_CSS + html + summary + TABLE_SPLIT_JS)

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

        self.section(title)

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
            html_content = (
                enhanced_css
                + NEWICK_TEMPLATE_COMBINED.format(newick1.strip(), newick2.strip())
                + NEWICK_HIGHLIGHT_JS
            )
        else:
            # Single tree case
            html_content = (
                enhanced_css
                + NEWICK_TEMPLATE_SINGLE_COMBINED.format(newick1.strip())
                + NEWICK_HIGHLIGHT_JS
            )

        self.raw_html(html_content)
