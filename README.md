# BranchArchitect

Brancharchitect implements algorithms to deal with tree trajectory.
Its main feature is an algorithm to identify a "jumping taxon", that is given two trees that differ by exactly one taxon, that taxon can efficiently be identified.
Additionally brancharchitect can read newick files, write json files, calculate the consensus tree and visualise trees as svgs.

# Exampels

## Parse Newick into Python Tree

```{python}
from brancharchitect.newick_parser import parse_newick

with open('newick_file.nwk') as f:
    tree = parse_newick(f.read())
```

## Serialize Tree into JSON


```{python}
from brancharchitect._io import read_newick, write_json

tree = read_newick('newick_file.nwk')
write_json(tree, 'json_file.json')
```

## Call jumping Taxa on a pair of trees

To find jumping taxa, both trees must have the same taxa and the taxa order must be identical.
To ensure that the taxa order is identical, either read both trees from the same file, provide an explicit taxa order or give the taxa order of the first tree when parsing the second tree.
The first example shows parsing both trees from the same file:

```{python}
from brancharchitect.io import read_newick
from brancharchitect.jumping_taxa import call_jumping_taxa

tree1, tree2 = read_newick('newick_file.nwk')

jumping_taxa = call_jumping_taxa(tree1, tree2)
```

Alternatively provide an explicit order:

```{python}
order = ['A', 'C', 'B', 'E']

tree1 = read_newick('newick_file1.nwk', order)
tree2 = read_newick('newick_file2.nwk', order)

jumping_taxa = call_jumping_taxa(tree1, tree2)
```

Or give the the order of the first tree when parsing the other tree:

```{python}
tree1 = read_newick('newick_file1.nwk')
tree2 = read_newick('newick_file2.nwk', tree1._order)

jumping_taxa = call_jumping_taxa(tree1, tree2)
```

## Generate SVG


```{python}

from brancharchitect.io import read_newick, write_svg

tree = read_newick('newick_file.nwk')
write_svg(tree, 'image.svg')

