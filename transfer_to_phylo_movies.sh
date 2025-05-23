#!/bin/bash

# Variables
BRANCH_ARCHITECT_DIR="/Users/berksakalli/Projects/BranchArchitect"  # Change this to the path where BranchArchitect is
PHYLO_MOVIES_BACKEND_DIR="/Users/berksakalli/Projects/phylo-movies/backend"  # New backend directory for PhyloMovies
DIST_DIR="$BRANCH_ARCHITECT_DIR/dist"
BACKUP_DIR="$PHYLO_MOVIES_BACKEND_DIR/brancharchitect_wheel_backups"

# --- REVERT FLAG SUPPORT ---
if [ "$1" = "--revert" ]; then
    REVERT_WHEEL="$2"
    if [ -z "$2" ]; then
        echo "Usage: $0 --revert <wheel_name>"
        exit 1
    fi
    if [ ! -e "$BACKUP_DIR/$2" ]; then
        echo "Backup wheel $2 not found in $BACKUP_DIR"
        exit 1
    fi
    echo "Reverting to $2 from backup..."
    cp "$BACKUP_DIR/$2" "$PHYLO_MOVIES_BACKEND_DIR/" || { echo "Failed to copy wheel."; exit 1; }
    cd "$PHYLO_MOVIES_BACKEND_DIR" || { echo "Directory not found: $PHYLO_MOVIES_BACKEND_DIR"; exit 1; }
    echo "Installing $2 in PhyloMovies backend..."
    poetry run pip install --force-reinstall "$2"
    echo "Revert complete!"
    cd "$BRANCH_ARCHITECT_DIR"
    exit 0
fi

# Step 0: Ensure backup directory exists
if [ ! -d "$BACKUP_DIR" ]; then
    echo "Creating backup directory at $BACKUP_DIR"
    mkdir -p "$BACKUP_DIR"
fi

# Step 1: Bump the version (minor)
cd "$BRANCH_ARCHITECT_DIR" || { echo "Directory not found: $BRANCH_ARCHITECT_DIR"; exit 1; }
echo "Bumping BranchArchitect version (minor)..."
poetry version minor

# Step 1b: Build the package
echo "Building BranchArchitect with Poetry..."
poetry build

# Step 2: Find the latest wheel file
WHEEL_FILE=$(ls -t "$DIST_DIR"/*.whl | head -n 1)
WHEEL_BASENAME=$(basename "$WHEEL_FILE")

# Step 3: Backup existing wheel(s) in PhyloMovies backend (if not already backed up)
for OLD_WHEEL in "$PHYLO_MOVIES_BACKEND_DIR"/*.whl; do
    [ -e "$OLD_WHEEL" ] || continue
    OLD_BASENAME=$(basename "$OLD_WHEEL")
    if [ ! -e "$BACKUP_DIR/$OLD_BASENAME" ]; then
        echo "Backing up $OLD_WHEEL to $BACKUP_DIR"
        mv "$OLD_WHEEL" "$BACKUP_DIR/"
    else
        echo "Backup for $OLD_BASENAME already exists, skipping move."
    fi
done

# Step 4: Copy the new wheel (do not overwrite if already present)
if [ -e "$PHYLO_MOVIES_BACKEND_DIR/$WHEEL_BASENAME" ]; then
    echo "Wheel $WHEEL_BASENAME already exists in $PHYLO_MOVIES_BACKEND_DIR, not overwriting."
else
    echo "Copying $WHEEL_FILE to PhyloMovies backend..."
    cp "$WHEEL_FILE" "$PHYLO_MOVIES_BACKEND_DIR/"
fi

# Step 5: Install the new wheel
cd "$PHYLO_MOVIES_BACKEND_DIR" || { echo "Directory not found: $PHYLO_MOVIES_BACKEND_DIR"; exit 1; }
echo "Installing $WHEEL_BASENAME in PhyloMovies backend..."
poetry run pip install --force-reinstall "$WHEEL_BASENAME"

echo "BranchArchitect has been built and installed in PhyloMovies backend!"

# Return to BranchArchitect directory
cd "$BRANCH_ARCHITECT_DIR"