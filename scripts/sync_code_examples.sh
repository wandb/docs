#!/bin/bash
# Sync ground truth code examples from docs-code-eval repo to docs snippets
# This script temporarily adds docs-code-eval as a submodule, copies the examples,
# then removes the submodule.

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DOCS_ROOT="$(dirname "$SCRIPT_DIR")"
SUBMODULE_PATH="$DOCS_ROOT/.temp_code_eval"

echo "🔄 Syncing code examples from docs-code-eval..."

# Optional: fine-grained or classic PAT with contents:read on wandb/docs-code-eval.
# Set by CI via secrets (see sync-code-examples workflow). Unset = anonymous clone (public repo only).
if [ -n "${DOCS_CODE_EVAL_READ_TOKEN:-}" ]; then
    EVAL_CLONE_URL="https://x-access-token:${DOCS_CODE_EVAL_READ_TOKEN}@github.com/wandb/docs-code-eval.git"
else
    EVAL_CLONE_URL="https://github.com/wandb/docs-code-eval.git"
fi

# Prefer an existing .temp_code_eval (e.g. local prep). In CI we clone after docs
# checkout with persist-credentials false; see sync-code-examples workflow comments.
if [ -d "$SUBMODULE_PATH/ground_truth" ] && compgen -G "$SUBMODULE_PATH/ground_truth/"*.py > /dev/null; then
    echo "   Using existing docs-code-eval checkout at $SUBMODULE_PATH"
else
    if [ -d "$SUBMODULE_PATH" ]; then
        echo "   Removing existing temporary directory..."
        rm -rf "$SUBMODULE_PATH"
    fi
    echo "   Cloning docs-code-eval repository..."
    if [ -n "${DOCS_CODE_EVAL_READ_TOKEN:-}" ]; then
        echo "   (HTTPS with PAT from DOCS_CODE_EVAL_READ_PAT)"
    else
        echo "   (anonymous HTTPS; set secret DOCS_CODE_EVAL_READ_PAT if the eval repo is private)"
    fi
    # - Clear extraheader so a stale Actions token does not override URL credentials.
    # - credential.helper= stops the runner's helper from prompting (CI has no TTY; you
    #   may otherwise see "could not read Username" / "No such device or address").
    export GIT_TERMINAL_PROMPT=0
    git \
        -c http.https://github.com/.extraheader= \
        -c credential.helper= \
        clone --depth 1 "$EVAL_CLONE_URL" "$SUBMODULE_PATH" --quiet
fi

# Copy Python files and create MDX wrappers
echo "   Copying Python code examples and creating MDX wrappers..."
PY_SNIPPETS_DIR="$DOCS_ROOT/snippets/en/_includes/code-examples"
mkdir -p "$PY_SNIPPETS_DIR"

# Copy Python files and create MDX wrappers
for pyfile in "$SUBMODULE_PATH/ground_truth/"*.py; do
    if [ -f "$pyfile" ]; then
        filename=$(basename "$pyfile")
        
        # Copy the Python file
        cp "$pyfile" "$PY_SNIPPETS_DIR/"
        
        # Create MDX wrapper
        mdxfile="$PY_SNIPPETS_DIR/${filename%.py}.mdx"
        cat > "$mdxfile" << EOF
\`\`\`python
$(cat "$pyfile")
\`\`\`
EOF
    fi
done

PY_COUNT=$(ls -1 "$PY_SNIPPETS_DIR"/*.py 2>/dev/null | wc -l | tr -d ' ')
echo "   ✓ Copied $PY_COUNT Python code examples and created MDX wrappers"

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
