import math
import json
import pytest
import logging
from pathlib import Path
from typing import Dict, List
from brancharchitect.jumping_taxa import call_jumping_taxa
from brancharchitect.newick_parser import parse_newick
from brancharchitect.tree import Node

logger = logging.getLogger()

test_data_list = []
i = 0

for p in Path("test/data/trees/").glob('*'):
    with open(p) as f:
        data = json.load(f)
        data['name'] = p.name
    if 'solutions' not in data:
        logger.warning(f'Testdata {p.name} does not have a solution')
    else:
        if data['solutions'] is None:
            data['solutions'] = [[]]
        else:
            data['solutions'] = [sorted([tuple(sorted(component)) for component in solution]) for solution in data['solutions']]
        test_data_list.append(data)

@pytest.fixture(params=test_data_list, ids=[test_data['name'] for test_data in test_data_list])
def test_data(request):
    return request.param

@pytest.mark.timeout(1)
def test_algorithm(test_data):
    n1 = test_data['tree1']
    n2 = test_data['tree2']
    solutions = test_data['solutions']
    comment = test_data['comment']

    t1 = parse_newick(n1)
    t2 = parse_newick(n2, t1._order)

    jt = call_jumping_taxa(t1, t2)
    jt = sorted([tuple(sorted(component)) for component in jt])
    assert jt in solutions
