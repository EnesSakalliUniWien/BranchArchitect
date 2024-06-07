# BranchArchitect

# usecase

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


```{python}
from brancharchitect._io import read_newick
from brancharchitect.jumping_taxa import call_jumping_taxa

tree1, tree2 = read_newick('newick_file.nwk')

jumping_taxa = call_jumping_taxa(tree1, tree2)
```
