from brancharchitect.parser.newick_parser import parse_newick
from brancharchitect.tree_interpolation.pair_interpolation import _unify_encodings


def test_unify_encodings_keeps_destination_split_index_available():
    source = parse_newick("((A:1,B:1):1,C:1);")
    destination = parse_newick("(A:1,(B:1,C:1):1);")

    _unify_encodings(source, destination)

    assert destination.taxa_encoding is source.taxa_encoding
    assert destination._split_index is not None
    assert destination.find_node_by_split(destination.split_indices) is destination
