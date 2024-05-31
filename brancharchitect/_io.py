from brancharchitect.newick_parser import parse_newick

# processed_tree_pairs contains the specified sequence for each adjacent pair
def transform_tree_from_file(file_name):
    newick_string = ""
    with open(file_name) as f:
        newick_string = f.readline()
    tree = parse_newick(newick_string)
    serialized_tree = serialize_to_dict_iterative(tree)
    f = open(f"{file_name}.json", "w")
    f.write(json.dumps(serialized_tree))
    f.close()
    return serialized_tree
