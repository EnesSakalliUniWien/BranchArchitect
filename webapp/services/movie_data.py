"""Movie data class for serializing backend responses to frontend format."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from brancharchitect.movie_pipeline.types import (
    InterpolationSequence,
    TreeMetadata as TreeMetadataType,
    TreePairSolution,
)
from brancharchitect.io import serialize_tree_list_to_json


@dataclass
class MovieData:
    """
    Data class that handles serialization of tree processing results
    for frontend consumption with hierarchical structure.
    """

    # Core tree data
    interpolated_trees: List[Dict[str, Any]]
    tree_metadata: List[TreeMetadataType]

    # Distance metrics
    rfd_list: List[float]
    weighted_robinson_foulds_distance_list: List[float]

    # Visualization data
    sorted_leaves: List[str]
    to_be_highlighted: List[Dict[str, Any]]
    split_change_tracking: List[Optional[List[int]]]
    subtree_tracking: List[Optional[List[int]]]

    # File and processing metadata
    file_name: str
    window_size: int
    window_step_size: int

    # MSA data
    msa_dict: Optional[Dict[str, str]]
    alignment_length: Optional[int]
    windows_are_overlapping: bool

    # Processing metadata
    original_tree_count: int
    interpolated_tree_count: int
    rooting_enabled: bool

    @classmethod
    def from_processing_result(
        cls,
        result: InterpolationSequence,
        filename: str,
        msa_data: Dict[str, Any],
        enable_rooting: bool,
        sorted_leaves: List[str],
    ) -> "MovieData":
        """
        Create MovieData from InterpolationSequence and additional data.

        Args:
            result: InterpolationSequence from TreeInterpolationPipeline
            filename: Original filename
            msa_data: Processed MSA data
            enable_rooting: Whether rooting was enabled
            sorted_leaves: Sorted leaf names from first tree

        Returns:
            MovieData instance ready for frontend serialization
        """

        # Additional debug: check the actual structure of interpolated_trees
        interpolated_trees = result["interpolated_trees"]

        serialized_trees = serialize_tree_list_to_json(interpolated_trees)
        rfd_list = result.get("rfd_list", [])
        wrfd_list = result.get("wrfd_list", [])

        # Store complete tree metadata for frontend
        tree_metadata = cls._process_tree_metadata(result["tree_metadata"])

        # Extract lattice edge tracking from tree metadata
        lattice_edge_tracking_data = cls._extract_s_edges_from_metadata(
            result["tree_metadata"]
        )

        # Extract subtree tracking from tree metadata
        subtree_tracking_data = cls._extract_subtrees_from_metadata(
            result["tree_metadata"]
        )

        return cls(
            interpolated_trees=serialized_trees,
            tree_metadata=tree_metadata,
            rfd_list=rfd_list,
            weighted_robinson_foulds_distance_list=wrfd_list,
            sorted_leaves=sorted_leaves,
            to_be_highlighted=cls._serialize_tree_pair_solutions(
                result["tree_pair_solutions"]
            ),
            split_change_tracking=lattice_edge_tracking_data,
            subtree_tracking=subtree_tracking_data,
            file_name=filename,
            window_size=msa_data.get("inferred_window_size", 1),
            window_step_size=msa_data.get("inferred_step_size", 1),
            msa_dict=msa_data.get("msa_dict"),
            alignment_length=msa_data.get("alignment_length"),
            windows_are_overlapping=msa_data.get("windows_are_overlapping", False),
            original_tree_count=result["original_tree_count"],
            interpolated_tree_count=result["interpolated_tree_count"],
            rooting_enabled=enable_rooting,
        )

    def to_frontend_dict(self) -> Dict[str, Any]:
        """
        Convert to flat dictionary format matching InterpolationSequence structure.

        Returns:s
            Dictionary with flattened structure matching InterpolationSequence from brancharchitect 0.59.0
        """
        result = {
            # Core flattened sequences - globally indexed (matches InterpolationSequence)
            "interpolated_trees": self.interpolated_trees,
            "tree_metadata": self.tree_metadata,
            # Tree pair solutions - keyed for easy lookup (matches InterpolationSequence)
            "tree_pair_solutions": self._extract_tree_pair_solutions_from_highlighted(),
            # Independent split change events per pair (0-based step ranges)
            "split_change_events": self._extract_split_change_events_from_highlighted(),
            # Global timeline mixing originals, split events, and explicit gaps
            "split_change_timeline": self._build_split_change_timeline(),
            "original_tree_count": self.original_tree_count,
            "interpolated_tree_count": self.interpolated_tree_count,
            # Additional frontend-specific data
            "sorted_leaves": self.sorted_leaves,
            # New name; keep legacy alias below for compatibility
            "split_change_tracking": self.split_change_tracking,
            "subtree_tracking": self.subtree_tracking,
            "covers": [],
            # MSA - simplified structure
            "msa": {
                "sequences": self.msa_dict,
                "alignment_length": self.alignment_length,
                "window_size": self.window_size,
                "step_size": self.window_step_size,
                "overlapping": self.windows_are_overlapping,
            },
            # File metadata - flattened
            "file_name": self.file_name,
            "processing_options": {
                "rooting_enabled": self.rooting_enabled,
            },
            "tree_count": {
                "original": self.original_tree_count,
                "interpolated": self.interpolated_tree_count,
            },
            "distances": {
                "robinson_foulds": self.rfd_list,
                "weighted_robinson_foulds": self.weighted_robinson_foulds_distance_list,
            },
            "to_be_highlighted": self.to_be_highlighted,
            "window_size": self.window_size,
            "window_step_size": self.window_step_size,
        }

        return result

    def _extract_split_change_events_from_highlighted(self) -> Dict[str, Any]:
        """Extract split_change_events per pair as an independent element.

        Returns a dict keyed by pair_key with value being the list of
        split_change_events already serialized to indices.
        """
        events: Dict[str, Any] = {}
        for solution in self.to_be_highlighted:
            if isinstance(solution, dict) and "pair_key" in solution:
                pair_key = solution["pair_key"]
                events[pair_key] = solution.get("split_change_events", [])
        return events

    def _build_split_change_timeline(self) -> List[Dict[str, Any]]:
        """Build a global timeline with originals, split events, and explicit gaps.

        Ordering:
        - T0
        - events for pair_0_1 (in order)
        - gap (original T1)
        - events for pair_1_2
        - gap (original T2)
        - ...

        Each event includes both local and global step ranges (0-based).
        Gaps reference the original tree element between pairs.
        """
        timeline: List[Dict[str, Any]] = []

        # Map pair_key -> first global index where this pair's interpolation starts
        first_global_for_pair: Dict[str, int] = {}
        # Map original tree index -> (global_index, name)
        originals: Dict[int, Dict[str, Any]] = {}

        for meta in self.tree_metadata:
            pair_key = meta.get("tree_pair_key")
            step = meta.get("step_in_pair")
            if pair_key and step == 1:
                # First step for this pair
                first_global_for_pair[pair_key] = meta["global_tree_index"]
            if pair_key is None and meta.get("source_tree_index") is not None:
                originals[meta["source_tree_index"]] = {
                    "global_index": meta["global_tree_index"],
                    "name": meta["tree_name"],
                }

        # Extract per-pair events already serialized to indices
        per_pair_events = self._extract_split_change_events_from_highlighted()

        # Construct timeline in natural pair order
        num_originals = self.original_tree_count
        for i in range(num_originals):
            # Add original tree T{i}
            if i in originals:
                orig = originals[i]
                timeline.append(
                    {
                        "type": "original",
                        "tree_index": i,
                        "global_index": orig["global_index"],
                        "name": orig["name"],
                    }
                )

            # Add events for pair i->i+1
            if i < num_originals - 1:
                pair_key = f"pair_{i}_{i + 1}"
                events = per_pair_events.get(pair_key, [])
                start_global = first_global_for_pair.get(pair_key)
                for ev in events:
                    step_start, step_end = ev.get("step_range", [0, -1])
                    g_start = (
                        None if start_global is None else start_global + step_start
                    )
                    g_end = None if start_global is None else start_global + step_end
                    timeline.append(
                        {
                            "type": "split_event",
                            "pair_key": pair_key,
                            "split": ev.get("split"),
                            "step_range_local": [step_start, step_end],
                            "step_range_global": [g_start, g_end],
                            "subtrees": ev.get("subtrees", []),
                        }
                    )

        return timeline

    @classmethod
    def _serialize_tree_pair_solutions(
        cls, tree_pair_solutions: Dict[str, TreePairSolution]
    ) -> List[Dict[str, Any]]:
        """Convert TreePairSolution objects to JSON-serializable format.

        Args:
            tree_pair_solutions: Dictionary mapping pair keys to TreePairSolution objects

        Returns:
            List of serialized solution dictionaries with Partition objects converted to lists
        """
        serialized_solutions: List[Dict[str, Any]] = []

        # tree_pair_solutions is now a dict with keys like "pair_0_1", "pair_1_2"
        for pair_key, solution in tree_pair_solutions.items():
            # Convert Partition objects to split indices (List[int]) for JSON serialization
            serialized_solution: Dict[str, Any] = {
                "pair_key": pair_key,
                "tree_indices": solution["tree_indices"],
                "lattice_edge_solutions": {
                    str(list(key.indices)): value
                    for key, value in solution["lattice_edge_solutions"].items()
                },
                "mapping_one": {
                    str(list(key.indices)): list(value.indices)
                    for key, value in solution["mapping_one"].items()
                },
                "mapping_two": {
                    str(list(key.indices)): list(value.indices)
                    for key, value in solution["mapping_two"].items()
                },
                "s_edge_sequence": [
                    list(edge.indices) if edge is not None else None
                    for edge in solution["s_edge_sequence"]
                ],
                "subtree_sequence": [
                    list(edge.indices) if edge is not None else None
                    for edge in solution.get("subtree_sequence", [])
                ],
                "s_edge_distances": {
                    str(list(key.indices)): value
                    for key, value in solution.get("s_edge_distances", {}).items()
                },
            }
            # Serialize split_change_events if present
            if "split_change_events" in solution:
                events = []
                for ev in solution["split_change_events"]:
                    events.append(
                        {
                            "split": list(ev["split"].indices),
                            "step_range": list(ev["step_range"]),  # already 0-based
                            "subtrees": [list(s.indices) for s in ev["subtrees"]],
                        }
                    )
                serialized_solution["split_change_events"] = events
            serialized_solutions.append(serialized_solution)

        return serialized_solutions

    @classmethod
    def _extract_s_edges_from_metadata(
        cls, tree_metadata: List[TreeMetadataType]
    ) -> List[Optional[List[int]]]:
        """Extract s_edge_tracker indices from tree metadata for lattice edge tracking.

        Args:
            tree_metadata: List of tree metadata from InterpolationSequence

        Returns:
            List of s_edge index lists (e.g., [1,2,3]) or None for each tree
        """
        return [meta.get("s_edge_tracker") for meta in tree_metadata]

    @classmethod
    def _extract_subtrees_from_metadata(
        cls, tree_metadata: List[TreeMetadataType]
    ) -> List[Optional[List[int]]]:
        """Extract subtree_tracker indices from tree metadata for subtree tracking.

        Args:
            tree_metadata: List of tree metadata from InterpolationSequence

        Returns:
            List of subtree index lists (e.g., [2,4,6]) or None for each tree
        """
        return [meta.get("subtree_tracker") for meta in tree_metadata]

    @classmethod
    def _process_tree_metadata(
        cls, tree_metadata: List[TreeMetadataType]
    ) -> List[TreeMetadataType]:
        """Process tree metadata to ensure JSON serialization and add phase information.

        Args:
            tree_metadata: List of tree metadata from InterpolationSequence

        Returns:
            List of processed metadata with added phase information
        """
        processed_metadata: List[TreeMetadataType] = []

        for meta in tree_metadata:
            # Create processed metadata preserving the TypedDict structure

            processed_meta: TreeMetadataType = TreeMetadataType(
                global_tree_index=meta["global_tree_index"],
                tree_name=meta["tree_name"],
                source_tree_index=meta["source_tree_index"],
                tree_pair_key=meta["tree_pair_key"],
                s_edge_tracker=meta.get("s_edge_tracker"),
                step_in_pair=meta.get("step_in_pair"),
                subtree_tracker=meta.get("subtree_tracker"),
            )

            processed_metadata.append(processed_meta)

        return processed_metadata

    def _extract_tree_pair_solutions_from_highlighted(self):
        """
        Extract tree_pair_solutions structure from highlighted elements.
        Returns a dict keyed by pair_key (e.g., "pair_0_1") with solution data.
        """
        # The to_be_highlighted field contains serialized tree_pair_solutions
        # Convert list format back to dict format for easier lookup
        tree_pair_solutions = {}
        for solution in self.to_be_highlighted:
            if isinstance(solution, dict) and "pair_key" in solution:
                pair_key = solution["pair_key"]
                tree_pair_solutions[pair_key] = solution

        return tree_pair_solutions

    @classmethod
    def create_empty(cls, filename: str) -> "MovieData":
        """Create empty MovieData for failed processing scenarios."""
        return cls(
            interpolated_trees=[],
            tree_metadata=[],
            rfd_list=[],
            weighted_robinson_foulds_distance_list=[],
            sorted_leaves=[],
            to_be_highlighted=[],
            split_change_tracking=[],
            subtree_tracking=[],
            file_name=filename,
            window_size=1,
            window_step_size=1,
            msa_dict=None,
            alignment_length=None,
            windows_are_overlapping=False,
            original_tree_count=0,
            interpolated_tree_count=0,
            rooting_enabled=False,
        )
