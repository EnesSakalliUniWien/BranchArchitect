#!/bin/bash

# Variables
BRANCH_ARCHITECT_DIR="/Users/berksakalli/Projects/BranchArchitect"  # Change this to the path where BranchArchitect is
PHYLO_MOVIES_DIR="/Users/berksakalli/Projects/phylo-movies"          # Change this to the path where PhyloMovies is
DIST_DIR="$BRANCH_ARCHITECT_DIR/dist"

# Step 1: Navigate to BranchArchitect directory and build the package using poetry
cd "$BRANCH_ARCHITECT_DIR" || { echo "Directory not found: $BRANCH_ARCHITECT_DIR"; exit 1; }
echo "Building BranchArchitect with Poetry..."
poetry build

# Step 2: Find the latest wheel file
WHEEL_FILE=$(ls -t "$DIST_DIR"/*.whl | head -n 1)

# Step 3: Move the wheel file to PhyloMovies project
echo "Moving $WHEEL_FILE to PhyloMovies..."
cp "$WHEEL_FILE" "$PHYLO_MOVIES_DIR"

# Step 4: Navigate to PhyloMovies directory and install the wheel using poetry
cd "$PHYLO_MOVIES_DIR" || { echo "Directory not found: $PHYLO_MOVIES_DIR"; exit 1; }
echo "Installing $WHEEL_FILE in PhyloMovies..."
poetry run pip install --force-reinstall "$WHEEL_FILE"

echo "BranchArchitect has been built and installed in PhyloMovies!"
cd "/Users/berksakalli/Projects/brancharchitect"          # Change this to the path where PhyloMovies is
