from brancharchitect.leaforder import tree_order_utils
from brancharchitect.parser.newick_parser import parse_newick


def test_classification_reuses_existing_split_indices(monkeypatch):
    reference = parse_newick("((A:1,B:1):1,C:1);")
    target = parse_newick("((A:1,B:1):1,C:1);", encoding=reference.taxa_encoding)

    original_initialize = type(reference).initialize_split_indices
    initialize_calls = 0

    def counted_initialize(self, encoding):
        nonlocal initialize_calls
        initialize_calls += 1
        return original_initialize(self, encoding)

    monkeypatch.setattr(type(reference), "initialize_split_indices", counted_initialize)

    tree_order_utils.classify_subtrees_using_set_ops(reference, target)

    assert initialize_calls == 0
