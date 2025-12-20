import urllib.request
from Bio import AlignIO, Phylo
import os
import sys


def prepare_dataset():
    # URL for the HIV-1 RT dataset (Nexus format)
    url = "https://raw.githubusercontent.com/veg/hyphy-analyses/master/alignment-viz/test/HIV_RT.nex"
    nexus_path = "datasets/HIV_RT.nex"
    fasta_path = "datasets/hiv_rt.fasta"
    newick_path = "datasets/hiv_rt.newick"

    # Ensure directory exists
    os.makedirs("datasets", exist_ok=True)

    if not os.path.exists(nexus_path):
        print(f"Downloading dataset from {url}...")
        try:
            urllib.request.urlretrieve(url, nexus_path)
            print(f"Downloaded to {nexus_path}")
        except Exception as e:
            print(f"Failed to download dataset: {e}")
            sys.exit(1)
    else:
        print(f"Dataset already exists at {nexus_path}")

    print("Converting Nexus alignment to FASTA...")
    try:
        count = AlignIO.convert(nexus_path, "nexus", fasta_path, "fasta")
        print(f"Successfully converted alignment to {fasta_path}")
    except Exception as e:
        print(f"Error converting alignment: {e}")

    print("Converting Nexus trees to Newick...")
    try:
        # Convert Nexus trees to Newick
        count = Phylo.convert(nexus_path, "nexus", newick_path, "newick")
        print(f"Successfully converted {count} trees to {newick_path}")

        if os.path.exists(newick_path) and os.path.getsize(newick_path) > 0:
            print(f"Tree file ready at: {os.path.abspath(newick_path)}")
        else:
            print("Error: Tree output file is empty or missing")
            sys.exit(1)

    except Exception as e:
        print(f"Error converting trees: {e}")
        sys.exit(1)


if __name__ == "__main__":
    prepare_dataset()
