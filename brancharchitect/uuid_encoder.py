import json
from typing import Any
from uuid import UUID


class UUIDEncoder(json.JSONEncoder):
    def default(self, o: Any):
        # Handle Partition objects
        if o.__class__.__name__ == "Partition":
            # Just return the indices as a list
            return list(o.indices)

        # Original UUID handling code
        if isinstance(o, UUID):
            return str(o)

        # Add handling for PartitionSet too
        if o.__class__.__name__ == "PartitionSet":
            # Return a list of lists of indices
            return [list(partition.indices) for partition in o]

        # Add handling for Node objects
        if o.__class__.__name__ == "Node":
            # Convert Node to a serializable dictionary
            node_dict = {
                "name": o.name,
                "length": o.length,
                "values": o.values,
                "children": o.children,  # This will recursively serialize child nodes
            }
            # Only include non-default/non-empty fields to keep JSON clean
            if hasattr(o, "split_indices") and o.split_indices:
                node_dict["split_indices"] = o.split_indices
            return node_dict

        return super().default(o)
