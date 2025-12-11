#!/bin/bash
# Sync ground truth code examples from docs-code-eval repo to docs snippets
# This script temporarily adds docs-code-eval as a submodule, copies the examples,
# then removes the submodule.

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DOCS_ROOT="$(dirname "$SCRIPT_DIR")"
SUBMODULE_PATH="$DOCS_ROOT/.temp_code_eval"
EVAL_REPO_URL="https://github.com/wandb/docs-code-eval.git"
TARGET_SNIPPETS_DIR="$DOCS_ROOT/snippets/code-examples"
CSV_FILE="llm_evaluation_tasks.csv"

echo "🔄 Syncing code examples from docs-code-eval..."

# Clean up any existing temp submodule
if [ -d "$SUBMODULE_PATH" ]; then
    echo "   Removing existing temporary directory..."
    rm -rf "$SUBMODULE_PATH"
fi

# Clone the repo (not as submodule, just a temp clone)
echo "   Cloning docs-code-eval repository..."
git clone --depth 1 "$EVAL_REPO_URL" "$SUBMODULE_PATH" --quiet

# Create target directory if it doesn't exist
mkdir -p "$TARGET_SNIPPETS_DIR"

# Create MDX wrappers from Python files (don't copy the Python files themselves)
echo "   Creating MDX snippet wrappers..."
MDX_SNIPPETS_DIR="$DOCS_ROOT/snippets/en/_includes/code-examples"
mkdir -p "$MDX_SNIPPETS_DIR"

for pyfile in "$SUBMODULE_PATH/ground_truth/"*.py; do
    if [ -f "$pyfile" ]; then
        filename=$(basename "$pyfile")
        basename="${filename%.py}"
        mdxfile="$MDX_SNIPPETS_DIR/${basename}.mdx"
        
        # Create MDX wrapper with code block
        cat > "$mdxfile" << EOF
\`\`\`python
$(cat "$pyfile")
\`\`\`
EOF
    fi
done

MDX_COUNT=$(ls -1 "$MDX_SNIPPETS_DIR"/*.mdx 2>/dev/null | wc -l | tr -d ' ')
echo "   ✓ Created $MDX_COUNT MDX snippet wrappers"

# Copy the CSV for metadata
echo "   Copying task metadata..."
cp "$SUBMODULE_PATH/$CSV_FILE" "$TARGET_SNIPPETS_DIR/"
echo "   ✓ Copied task metadata CSV"

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
