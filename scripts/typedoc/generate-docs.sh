#!/bin/bash

# TypeDoc Documentation Generator for W&B Query Panel
# This script would generate documentation from the actual W&B core source

set -e

echo "======================================================"
echo "W&B Query Panel Documentation Generator (TypeDoc)"
echo "======================================================"
echo ""

# Check if source path is provided
if [ -z "$1" ]; then
    echo "❌ Error: Please provide path to wandb/core repository"
    echo ""
    echo "Usage:"
    echo "  ./generate-docs.sh /path/to/wandb/core"
    echo ""
    echo "Example:"
    echo "  ./generate-docs.sh ~/repos/wandb-core"
    echo ""
    echo "This script will:"
    echo "  1. Use TypeDoc to generate documentation from TypeScript source"
    echo "  2. Output markdown files compatible with Hugo"
    echo "  3. Post-process for proper formatting"
    echo ""
    echo "Prerequisites:"
    echo "  - wandb/core repository with TypeScript source files"
    echo "  - TSDoc comments in the source code"
    echo ""
    exit 1
fi

WANDB_CORE_PATH="$1"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUTPUT_DIR="$SCRIPT_DIR/../../content/en/ref/query-panel"

echo "📍 Source: $WANDB_CORE_PATH"
echo "📍 Output: $OUTPUT_DIR"
echo ""

# Verify source exists
if [ ! -d "$WANDB_CORE_PATH/frontends/weave/src/core" ]; then
    echo "❌ Error: Cannot find source at $WANDB_CORE_PATH/frontends/weave/src/core"
    echo "Please check the path to wandb/core repository"
    exit 1
fi

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing TypeDoc dependencies..."
    npm install
    echo ""
fi

# Clean previous output
if [ -d "$OUTPUT_DIR" ]; then
    echo "🧹 Cleaning previous documentation..."
    rm -rf "$OUTPUT_DIR"
fi

# Update TypeDoc config with actual source path
echo "🔧 Configuring TypeDoc..."
sed -i.bak "s|PATH_TO_WANDB_CORE|$WANDB_CORE_PATH|g" typedoc.json
sed -i.bak "s|PATH_TO_WANDB_CORE|$WANDB_CORE_PATH|g" tsconfig.json

# Generate documentation
echo "📝 Generating documentation with TypeDoc..."
npx typedoc \
  --entryPoints "$WANDB_CORE_PATH/frontends/weave/src/core/ops/*.ts" \
  --out "$OUTPUT_DIR"

# Restore config files
mv typedoc.json.bak typedoc.json
mv tsconfig.json.bak tsconfig.json

# Post-process for Hugo
echo "🔧 Post-processing for Hugo..."
node postprocess-hugo.js

# Create index file
echo "📄 Creating index file..."
cat > "$OUTPUT_DIR/_index.md" << 'EOF'
---
title: Query Expression Language
menu:
    reference:
        identifier: query-panel
        parent: reference
        weight: 4
cascade:
    menu:
        reference:
            parent: query-panel
---

# W&B Query Expression Language

Use the query expressions to select and aggregate data across runs and projects.

## Overview

This documentation was automatically generated from the TypeScript source code using TypeDoc.

### Features

- Complete API documentation with examples
- Type-safe operation definitions
- Comprehensive parameter descriptions
- Cross-referenced data types

## See Also

- [Query Panels Guide](/guides/models/app/features/panels/query-panels/)
- [W&B Python SDK](/ref/python/)
EOF

echo ""
echo "✅ Documentation generated successfully!"
echo ""
echo "📍 Output location: $OUTPUT_DIR"
echo ""
echo "To view in Hugo:"
echo "  cd $(dirname $SCRIPT_DIR)"
echo "  hugo server"
echo ""
echo "Then navigate to: /ref/query-panel-generated/"
