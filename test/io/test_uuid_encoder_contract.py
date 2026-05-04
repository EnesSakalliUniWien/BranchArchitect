import json

from brancharchitect.io import UUIDEncoder
from brancharchitect.tree import Node


def test_uuid_encoder_serializes_empty_split_indices():
    encoding = {"A": 0}
    node = Node(name="root", taxa_encoding=encoding, split_indices=())

    serialized = json.loads(json.dumps(node, cls=UUIDEncoder))

    assert serialized["split_indices"] == []
