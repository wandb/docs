#!/bin/bash
# Sync ground truth code examples from docs-code-eval repo to docs snippets
# This script temporarily adds docs-code-eval as a submodule, copies the examples,
# then removes the submodule.

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DOCS_ROOT="$(dirname "$SCRIPT_DIR")"
SUBMODULE_PATH="$DOCS_ROOT/.temp_code_eval"
EVAL_REPO_URL="https://github.com/wandb/docs-code-eval.git"

echo "🔄 Syncing code examples from docs-code-eval..."

# Clean up any existing temp submodule
if [ -d "$SUBMODULE_PATH" ]; then
    echo "   Removing existing temporary directory..."
    rm -rf "$SUBMODULE_PATH"
fi

# Clone the repo (not as submodule, just a temp clone)
echo "   Cloning docs-code-eval repository..."
git clone --depth 1 "$EVAL_REPO_URL" "$SUBMODULE_PATH" --quiet

# Copy Python files directly
echo "   Copying Python code examples..."
PY_SNIPPETS_DIR="$DOCS_ROOT/snippets/en/_includes/code-examples"
mkdir -p "$PY_SNIPPETS_DIR"

# Copy Python files
cp "$SUBMODULE_PATH/ground_truth/"*.py "$PY_SNIPPETS_DIR/"

PY_COUNT=$(ls -1 "$PY_SNIPPETS_DIR"/*.py 2>/dev/null | wc -l | tr -d ' ')
echo "   ✓ Copied $PY_COUNT Python code examples"

# Generate CodeSnippet component with all imports
echo "   Generating CodeSnippet component..."
python3 "$SCRIPT_DIR/generate_code_snippet_component.py"

# Clean up
echo "   Cleaning up temporary clone..."
rm -rf "$SUBMODULE_PATH"

# Generate the cheat sheet page
echo "   Generating cheat sheet page..."
python3 "$SCRIPT_DIR/generate_cheat_sheet.py"

echo "✅ Sync complete!"
echo ""
echo "Next steps:"
echo "  1. Review the generated MDX wrappers in: snippets/en/_includes/code-examples/"
echo "  2. Review the cheat sheet pages: models/ref/sdk-coding-cheat-sheet/"
echo "  3. Commit the changes"
