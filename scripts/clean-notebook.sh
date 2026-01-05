#!/bin/bash
set -e

# Script to clean Jupyter notebook metadata using nb-clean
# This removes execution counts, outputs, and cell metadata that causes noisy diffs

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print error messages
error() {
  echo -e "${RED}Error:${NC} $1" >&2
  exit 1
}

# Function to print success messages
success() {
  echo -e "${GREEN}✓${NC} $1"
}

# Function to print info messages
info() {
  echo -e "${YELLOW}→${NC} $1"
}

# Check if nb-clean is installed (try both command and Python module)
NB_CLEAN_CMD=""
if command -v nb-clean &> /dev/null; then
  NB_CLEAN_CMD="nb-clean"
elif python3 -c "import nb_clean" &> /dev/null; then
  NB_CLEAN_CMD="python3 -m nb_clean"
else
  error "nb-clean is not installed.

To install nb-clean, run:
  pip install nb-clean

Or with pipx (recommended for CLI tools):
  pipx install nb-clean

For more information, see: https://github.com/srstevenson/nb-clean"
fi

# Check if a notebook path was provided
if [ $# -eq 0 ]; then
  error "No notebook path provided.

Usage:
  $0 <path-to-notebook.ipynb>

Example:
  $0 weave/cookbooks/source/my-notebook.ipynb"
fi

NOTEBOOK_PATH="$1"

# Check if the file exists
if [ ! -f "$NOTEBOOK_PATH" ]; then
  error "File not found: $NOTEBOOK_PATH"
fi

# Check if it's a .ipynb file
if [[ ! "$NOTEBOOK_PATH" =~ \.ipynb$ ]]; then
  error "File must be a Jupyter notebook (.ipynb): $NOTEBOOK_PATH"
fi

# Clean the notebook
info "Cleaning notebook: $NOTEBOOK_PATH"
$NB_CLEAN_CMD clean "$NOTEBOOK_PATH" --preserve-cell-metadata tags

# Check if the cleaning was successful
if [ $? -eq 0 ]; then
  success "Notebook cleaned successfully"
  echo ""
  echo "The following were removed:"
  echo "  - Execution counts"
  echo "  - Cell outputs"
  echo "  - Cell metadata (except 'tags')"
  echo "  - Notebook metadata (except essential fields)"
  echo ""
  echo "You can now commit this cleaned notebook."
else
  error "Failed to clean notebook"
fi
