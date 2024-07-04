from brancharchitect.io import read_newick, write_json
from brancharchitect.newick_parser import parse_newick
from brancharchitect.svg import generate_svg
from brancharchitect.jumping_taxa.deletion_algorithm import get_child
import json

import tempfile


def test_read_newick_write_json():
    newick = "((A[value=3],(B,C),((D,E),((F,G),H))),I);"

    with tempfile.NamedTemporaryFile(mode='w') as f_in:
        f_in.write(newick)
        f_in.flush()
        tree = read_newick(f_in.name)

    assert len(tree.children) == 2
    assert len(get_child(tree, 0).children) == 3

    assert get_child(tree, 0, 0).name == "A"
    assert len(get_child(tree, 0, 0).children) == 0

    assert get_child(tree, 0, 1, 0).name == "B"
    assert get_child(tree, 0, 1, 1).name == "C"
    assert get_child(tree, 0, 2, 0, 0).name == "D"
    assert get_child(tree, 0, 2, 0, 1).name == "E"
    assert get_child(tree, 0, 2, 1, 0, 0).name == "F"
    assert get_child(tree, 0, 2, 1, 0, 1).name == "G"
    assert get_child(tree, 0, 2, 1, 1).name == "H"
    assert get_child(tree, 1).name == "I"


    with tempfile.NamedTemporaryFile(mode='r') as f_out:
        write_json(tree, f_out.name)
        json_data = f_out.read()

    tree2 = json.loads(json_data)

    assert get_child(tree2, 0, 1, 0)['name'] == "B"
    assert get_child(tree2, 0, 1, 1)['name'] == "C"
    assert get_child(tree2, 0, 2, 0, 0)['name'] == "D"
    assert get_child(tree2, 0, 2, 0, 1)['name'] == "E"
    assert get_child(tree2, 0, 2, 1, 0, 0)['name'] == "F"
    assert get_child(tree2, 0, 2, 1, 0, 1)['name'] == "G"
    assert get_child(tree2, 0, 2, 1, 1)['name'] == "H"
    assert get_child(tree2, 1)['name'] == "I"


def test_generate_svg():
    newick = "((A[value=3],(B,C),((D,E),((F,G),H))),I);"
    tree = parse_newick(newick)
    svg = generate_svg(tree, 100)
