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
    tree_pair_solutions: Dict[str, TreePairSolution]
    split_change_tracking: List[Optional[List[int]]]

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

        # Derive split_change_tracking from split_change_events and metadata alignment
        events_by_pair = cls._events_from_solutions(result["tree_pair_solutions"])
        lattice_edge_tracking_data = cls._derive_split_change_tracking_from_events(
            tree_metadata, events_by_pair
        )

        return cls(
            interpolated_trees=serialized_trees,
            tree_metadata=tree_metadata,
            rfd_list=rfd_list,
            weighted_robinson_foulds_distance_list=wrfd_list,
            sorted_leaves=sorted_leaves,
            tree_pair_solutions=result["tree_pair_solutions"],
            split_change_tracking=lattice_edge_tracking_data,
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

        Returns:
            Dictionary with flattened structure matching InterpolationSequence from brancharchitect 0.59.0
        """
        result: Dict[str, Any] = {
            # Core flattened sequences - globally indexed (matches InterpolationSequence)
            "interpolated_trees": self.interpolated_trees,
            "tree_metadata": self.tree_metadata,
            # Tree pair solutions - keyed for easy lookup (matches InterpolationSequence)
            "tree_pair_solutions": self._serialize_tree_pair_solutions(
                self.tree_pair_solutions
            ),
            # Independent split change events per pair (0-based step ranges)
            "split_change_events": self._extract_split_change_events_from_solutions(),
            # Global timeline mixing originals, split events, and explicit gaps
            "split_change_timeline": self._build_split_change_timeline(),
            "original_tree_count": self.original_tree_count,
            "interpolated_tree_count": self.interpolated_tree_count,
            # Additional frontend-specific data
            "sorted_leaves": self.sorted_leaves,
            # New name; keep legacy alias below for compatibility
            "split_change_tracking": self.split_change_tracking,
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
            # 'to_be_highlighted' removed; use 'tree_pair_solutions' instead
            "window_size": self.window_size,
            "window_step_size": self.window_step_size,
        }

        return result

    @classmethod
    def _events_from_solutions(
        cls, tree_pair_solutions: Dict[str, TreePairSolution]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Build serialized split_change_events per pair from raw solutions.

        Returns a dict keyed by pair_key with list of events, where each event
        item contains "split": List[int] and "step_range": List[int].
        """
        events: Dict[str, List[Dict[str, Any]]] = {}
        for pair_key, solution in tree_pair_solutions.items():
            pair_events = solution.get("split_change_events", [])  # type: ignore[arg-type]
            serialized_events: List[Dict[str, Any]] = []
            for ev in pair_events:
                serialized_events.append(
                    {
                        "split": list(ev["split"].indices),
                        "step_range": list(ev["step_range"]),
                    }
                )
            events[pair_key] = serialized_events
        return events

    @classmethod
    def _derive_split_change_tracking_from_events(
        cls,
        processed_tree_metadata: List[TreeMetadataType],
        events_by_pair: Dict[str, List[Dict[str, Any]]],
    ) -> List[Optional[List[int]]]:
        """Derive per-tree split tracking aligned to metadata indices.

        For each interpolation pair, compute the first global index where the
        pair starts (step_in_pair == 1). Then, for each split_change_event in
        that pair, mark all steps in its local step_range by assigning the
        event's split at the corresponding global indices.

        Originals (trees with tree_pair_key is None) remain as None values.
        """
        tracking: List[Optional[List[int]]] = [None for _ in processed_tree_metadata]

        # Index first global index per pair (where local step 0 aligns)
        first_global_for_pair: Dict[str, int] = {}
        for meta in processed_tree_metadata:
            pair_key = meta.get("tree_pair_key")
            step = meta.get("step_in_pair")
            if pair_key and step == 1:
                first_global_for_pair[pair_key] = meta["global_tree_index"]

        # Fill tracking using events and computed anchors
        for pair_key, events in events_by_pair.items():
            start_global = first_global_for_pair.get(pair_key)
            if start_global is None:
                continue
            for ev in events:
                step_start, step_end = ev.get("step_range", [0, -1])
                split = ev.get("split")
                if split is None:
                    continue
                # Assign inclusive range [step_start, step_end]
                for local_step in range(step_start, step_end + 1):
                    idx = start_global + local_step
                    if 0 <= idx < len(tracking):
                        tracking[idx] = split

        return tracking

    def _extract_split_change_events_from_solutions(
        self,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Extract split_change_events per pair as an independent element.

        Returns a dict keyed by pair_key with value being the list of
        split_change_events already serialized to indices.
        """
        events: Dict[str, List[Dict[str, Any]]] = {}
        for pair_key, solution in self.tree_pair_solutions.items():
            pair_events = solution.get("split_change_events", [])  # type: ignore[arg-type]
            # Serialize Partition objects to index lists
            serialized_events: List[Dict[str, Any]] = []
            for ev in pair_events:
                serialized_events.append(
                    {
                        "split": list(ev["split"].indices),
                        "step_range": list(ev["step_range"]),
                    }
                )
            events[pair_key] = serialized_events
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

        first_global_for_pair, originals = self._index_timeline_anchors()
        per_pair_events = self._extract_split_change_events_from_solutions()

        # Construct timeline in natural pair order
        for i in range(self.original_tree_count):
            self._append_original_entry(timeline, i, originals)
            if i < self.original_tree_count - 1:
                pair_key = f"pair_{i}_{i + 1}"
                events = per_pair_events.get(pair_key, [])
                start_global = first_global_for_pair.get(pair_key)
                self._append_pair_events(timeline, pair_key, events, start_global)

        return timeline

    def _index_timeline_anchors(
        self,
    ) -> tuple[Dict[str, int], Dict[int, Dict[str, Any]]]:
        """Index first-step globals per pair and originals with their global indices.

        Returns:
            Tuple of (first_global_for_pair, originals_by_index)
        """
        first_global_for_pair: Dict[str, int] = {}
        originals: Dict[int, Dict[str, Any]] = {}

        # Determine originals by order of appearance with no pair_key
        orig_counter = 0
        for meta in self.tree_metadata:
            pair_key = meta.get("tree_pair_key")
            step = meta.get("step_in_pair")
            if pair_key and step == 1:
                first_global_for_pair[pair_key] = meta["global_tree_index"]
            if pair_key is None:
                originals[orig_counter] = {
                    "global_index": meta["global_tree_index"],
                    "name": "",
                }
                orig_counter += 1

        return first_global_for_pair, originals

    def _append_original_entry(
        self,
        timeline: List[Dict[str, Any]],
        tree_index: int,
        originals: Dict[int, Dict[str, Any]],
    ) -> None:
        """Append the original tree entry to the timeline if available."""
        if tree_index not in originals:
            return
        orig = originals[tree_index]
        timeline.append(
            {
                "type": "original",
                "tree_index": tree_index,
                "global_index": orig["global_index"],
                "name": orig.get("name", ""),
            }
        )

    def _append_pair_events(
        self,
        timeline: List[Dict[str, Any]],
        pair_key: str,
        events: List[Dict[str, Any]],
        start_global: Optional[int],
    ) -> None:
        """Append split events for one pair, computing global ranges from the start index."""
        if not events:
            return
        for ev in events:
            step_start, step_end = ev.get("step_range", [0, -1])
            g_start = None if start_global is None else start_global + step_start
            g_end = None if start_global is None else start_global + step_end
            timeline.append(
                {
                    "type": "split_event",
                    "pair_key": pair_key,
                    "split": ev.get("split"),
                    "step_range_local": [step_start, step_end],
                    "step_range_global": [g_start, g_end],
                }
            )

    @classmethod
    def _serialize_tree_pair_solutions(
        cls, tree_pair_solutions: Dict[str, TreePairSolution]
    ) -> Dict[str, Dict[str, Any]]:
        """Convert TreePairSolution objects to a JSON-serializable dict keyed by pair_key."""
        serialized: Dict[str, Dict[str, Any]] = {}
        for pair_key, solution in tree_pair_solutions.items():
            item: Dict[str, Any] = {
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
                "ancestor_of_changing_splits": [
                    list(edge.indices) if edge is not None else None
                    for edge in solution["ancestor_of_changing_splits"]
                ],
                # subtree_sequence removed
            }
            if "split_change_events" in solution:
                events_ser: list[dict[str, Any]] = []
                for ev in solution["split_change_events"]:
                    events_ser.append(
                        {
                            "split": list(ev["split"].indices),
                            "step_range": list(ev["step_range"]),
                            # subtrees removed
                        }
                    )
                item["split_change_events"] = events_ser
            serialized[pair_key] = item
        return serialized

    @classmethod
    def _extract_s_edges_from_metadata(
        cls, tree_metadata: List[TreeMetadataType]
    ) -> List[Optional[List[int]]]:
        """Compatibility stub: s_edge_tracker removed from metadata.

        Returns a list of None values aligned with tree_metadata length.
        """
        return [None for _ in tree_metadata]

    @classmethod

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
            processed_metadata.append(
                TreeMetadataType(
                    global_tree_index=meta["global_tree_index"],
                    tree_pair_key=meta.get("tree_pair_key"),
                    step_in_pair=meta.get("step_in_pair"),
                )
            )

        return processed_metadata

    # Removed: _extract_tree_pair_solutions_from_highlighted (use self.tree_pair_solutions)

    @classmethod
    def create_empty(cls, filename: str) -> "MovieData":
        """Create empty MovieData for failed processing scenarios."""
        return cls(
            interpolated_trees=[],
            tree_metadata=[],
            rfd_list=[],
            weighted_robinson_foulds_distance_list=[],
            sorted_leaves=[],
            tree_pair_solutions={},
            split_change_tracking=[],
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
