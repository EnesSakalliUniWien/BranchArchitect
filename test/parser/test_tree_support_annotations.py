from brancharchitect.parser import parse_newick


def _find_split(node_dict, split):
    if node_dict["split_indices"] == split:
        return node_dict
    for child in node_dict["children"]:
        found = _find_split(child, split)
        if found is not None:
            return found
    return None


def test_numeric_internal_label_serializes_as_branch_support_annotation():
    tree = parse_newick("((A:1,B:1)95:2,C:3);")

    ab_node = _find_split(tree.to_dict(), [0, 1])
    fields = ab_node["annotations"]["fields"]

    assert ab_node["name"] == ""
    assert fields["label.raw_internal"]["value"] == "95"
    assert fields["support.bootstrap.value"] == {
        "path": ["support", "bootstrap", "value"],
        "label": "Bootstrap",
        "value": 95.0,
        "value_type": "number",
        "role": "branch_support",
        "unit": "percent",
        "analysis": {
            "type": "tree_inference",
            "method": "bootstrap",
        },
    }


def test_slash_internal_label_serializes_iqtree_support_annotation():
    tree = parse_newick("((A:1,B:1)88.5/99:2,C:3);")

    ab_node = _find_split(tree.to_dict(), [0, 1])
    fields = ab_node["annotations"]["fields"]

    assert fields["label.raw_internal"]["value"] == "88.5/99"
    assert fields["support.iqtree.sh_alrt"]["value"] == 88.5
    assert fields["support.iqtree.sh_alrt"]["analysis"]["method"] == "iqtree"
    assert fields["support.iqtree.ufboot"]["value"] == 99.0
    assert fields["support.iqtree.ufboot"]["analysis"]["mode"] == "sh_alrt_ufboot"


def test_iqtree_single_value_support_kind_uses_canonical_iqtree_annotation():
    tree = parse_newick("((A:1,B:1)95[support_kind=ufboot]:2,C:3);")

    ab_node = _find_split(tree.to_dict(), [0, 1])
    fields = ab_node["annotations"]["fields"]

    assert "support.bootstrap.value" not in fields
    assert fields["support.iqtree.ufboot"] == {
        "path": ["support", "iqtree", "ufboot"],
        "label": "UFBoot",
        "value": 95.0,
        "value_type": "number",
        "role": "branch_support",
        "unit": "percent",
        "analysis": {
            "type": "tree_inference",
            "method": "iqtree",
            "mode": "ufboot",
        },
    }


def test_iqtree_sh_alrt_single_value_support_kind_uses_canonical_annotation():
    tree = parse_newick("((A:1,B:1)88.5[support_kind=sh_alrt]:2,C:3);")

    ab_node = _find_split(tree.to_dict(), [0, 1])
    fields = ab_node["annotations"]["fields"]

    assert "support.bootstrap.value" not in fields
    assert fields["support.iqtree.sh_alrt"]["label"] == "SH-aLRT"
    assert fields["support.iqtree.sh_alrt"]["value"] == 88.5
    assert fields["support.iqtree.sh_alrt"]["analysis"] == {
        "type": "tree_inference",
        "method": "iqtree",
        "mode": "sh_alrt",
    }


def test_metadata_support_serializes_without_changing_leaf_names():
    tree = parse_newick("((A:1,B:1)[support=72.5]:2,C:3);")

    payload = tree.to_dict()
    ab_node = _find_split(payload, [0, 1])
    a_node = _find_split(payload, [0])
    fields = ab_node["annotations"]["fields"]

    assert a_node["name"] == "A"
    assert "label.raw_internal" not in fields
    assert fields["support.metadata.support"]["value"] == 72.5
    assert fields["support.metadata.support"]["role"] == "branch_support"


def test_unannotated_nodes_omit_annotations_container():
    tree = parse_newick("((A:1,B:1):2,C:3);")

    payload = tree.to_dict()
    ab_node = _find_split(payload, [0, 1])
    a_node = _find_split(payload, [0])

    assert "annotations" not in payload
    assert "annotations" not in ab_node
    assert "annotations" not in a_node


def test_numeric_label_can_carry_rogue_taxa_split_frequency_metadata():
    tree = parse_newick(
        "((A:1,B:1)87.5[support_kind=bootstrap_replicate_split_frequency,"
        "replicate_count=175,replicate_total=200]:2,C:3);"
    )

    ab_node = _find_split(tree.to_dict(), [0, 1])
    fields = ab_node["annotations"]["fields"]

    assert fields["label.raw_internal"]["value"] == "87.5"
    assert fields["support.bootstrap_rogue.frequency"]["value"] == 87.5
    assert fields["support.bootstrap_rogue.frequency"]["analysis"] == {
        "type": "rogue_taxa",
        "method": "bootstrap_replicate_split_frequency",
    }
    assert fields["support.bootstrap_rogue.replicate_count"]["value"] == 175.0
    assert fields["support.bootstrap_rogue.replicate_total"]["value"] == 200.0


def test_numeric_label_can_carry_rogue_taxa_subtree_frequency_metadata():
    tree = parse_newick(
        "((A:1,B:1)87.5[support_kind=bootstrap_replicate_subtree_frequency,"
        "bootstrap_frequency=87.5,replicate_count=175,replicate_total=200]:2,C:3);"
    )

    ab_node = _find_split(tree.to_dict(), [0, 1])
    fields = ab_node["annotations"]["fields"]

    assert fields["label.raw_internal"]["value"] == "87.5"
    assert fields["support.bootstrap_rogue.frequency"]["label"] == "Bootstrap Subtree Frequency"
    assert fields["support.bootstrap_rogue.frequency"]["value"] == 87.5
    assert fields["support.bootstrap_rogue.frequency"]["analysis"] == {
        "type": "rogue_taxa",
        "method": "bootstrap_replicate_subtree_frequency",
    }
    assert fields["support.bootstrap_rogue.replicate_count"]["value"] == 175.0
    assert fields["support.bootstrap_rogue.replicate_total"]["value"] == 200.0


def test_metadata_can_carry_rogue_taxa_split_frequency_without_internal_label():
    tree = parse_newick(
        "((A:1,B:1)[support_kind=bootstrap_replicate_subtree_frequency,"
        "bootstrap_frequency=87.5,replicate_count=175,replicate_total=200]:2,C:3);"
    )

    ab_node = _find_split(tree.to_dict(), [0, 1])
    fields = ab_node["annotations"]["fields"]

    assert "label.raw_internal" not in fields
    assert fields["support.bootstrap_rogue.frequency"]["label"] == "Bootstrap Subtree Frequency"
    assert fields["support.bootstrap_rogue.frequency"]["value"] == 87.5
    assert fields["support.bootstrap_rogue.frequency"]["analysis"]["method"] == "bootstrap_replicate_subtree_frequency"
    assert fields["support.bootstrap_rogue.replicate_count"]["value"] == 175.0
    assert fields["support.bootstrap_rogue.replicate_total"]["value"] == 200.0
