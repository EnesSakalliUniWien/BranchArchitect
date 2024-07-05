from brancharchitect.io import read_newick, write_svg
from brancharchitect.newick_parser import parse_newick
import click
from pathlib import Path
import json

def read_file(path, newick):
    if newick:
        return read_newick(path)
    else:
        with open(path) as f:
            data = json.load(f)
        t1 = parse_newick(data['tree1'])
        t2 = parse_newick(data['tree2'], order=t1._order)
        return [t1, t2]

@click.command()
@click.argument('path')
@click.argument('out')
@click.option('--newick/--testdata', default=True)
def export(path, out, newick):

    if Path(out).exists():
        out = str(Path(out) / '{0}.svg')

    for i, tree in enumerate(read_file(path, newick)):
        write_svg(tree, out.format(str(i)), ignore_branch_lengths=True)

if __name__ == '__main__':
    export()
