from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import json

from brancharchitect.tree import Node
from brancharchitect.io import read_newick, UUIDEncoder


@dataclass
class Tree:
    """
    A wrapper around a phylogenetic tree (Node) with additional metadata.
    Does not replace Node, but strictly types a 'Tree' entity in a list.
    """

    root: Node
    index: Optional[int] = None
    name: Optional[str] = None
    taxa_encoding: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if not self.taxa_encoding and self.root.taxa_encoding:
            self.taxa_encoding = self.root.taxa_encoding
        if not self.name and self.root.name:
            self.name = self.root.name

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the tree structure to a dictionary."""
        return self.root.to_dict()

    def get_newick(self) -> str:
        """Get Newick string representation."""
        return self.root.to_newick()


@dataclass
class TreeList:
    """
    A collection of trees sharing a common set of taxa.
    Enforces consistency across the batch.
    """

    trees: List[Tree]
    taxa: List[str]

    @property
    def encoding(self) -> Dict[str, int]:
        return {name: i for i, name in enumerate(self.taxa)}

    @classmethod
    def from_newick(
        cls, path: str, force_list: bool = True, treat_zero_as_epsilon: bool = True
    ) -> "TreeList":
        """
        Factory method to create a TreeList from a Newick file.
        """
        nodes_or_node = read_newick(
            path, force_list=force_list, treat_zero_as_epsilon=treat_zero_as_epsilon
        )

        # Ensure we have a list of Nodes
        nodes: List[Node] = (
            nodes_or_node if isinstance(nodes_or_node, list) else [nodes_or_node]
        )

        if not nodes:
            return cls(trees=[], taxa=[])

        # Extract consensus taxa from the first tree (assuming consistency as per io.py rules)
        taxa = list(nodes[0].get_current_order())

        # Wrap Nodes in Tree objects
        wrapped_trees = [
            Tree(root=node, index=i, name=f"Tree_{i}") for i, node in enumerate(nodes)
        ]

        return cls(trees=wrapped_trees, taxa=taxa)

    def to_json(self) -> str:
        """
        Serialize the entire TreeList to a JSON string.
        """
        # Unwrap to list of dicts as expected by frontend
        data = [t.root.to_dict() for t in self.trees]
        return json.dumps(data, cls=UUIDEncoder)

    def __len__(self) -> int:
        return len(self.trees)

    def __getitem__(self, index: int) -> Tree:
        return self.trees[index]

    def __iter__(self):
        return iter(self.trees)
