#!/bin/bash
# Test script for generating a single SDK version release note

VERSION=${1:-0.21.1}

echo "Testing SDK release note generation for version $VERSION"
echo "================================================"

# Check if wandb repo exists locally
WANDB_PATH="/Users/matt.linville/wandb/CHANGELOG.md"
if [ ! -f "$WANDB_PATH" ]; then
    echo "❌ Error: CHANGELOG.md not found at $WANDB_PATH"
    echo "Please ensure the wandb repo is available"
    exit 1
fi

# Create a test output directory
TEST_OUTPUT="./test-sdk-output"
mkdir -p "$TEST_OUTPUT"

# Run the script for a single version
python3 scripts/sdk-changelog-to-hugo.py \
    --changelog "$WANDB_PATH" \
    --output "$TEST_OUTPUT" \
    --version "$VERSION"

# Check if file was created
OUTPUT_FILE="$TEST_OUTPUT/${VERSION}.md"
if [ -f "$OUTPUT_FILE" ]; then
    echo ""
    echo "✅ Success! File created: $OUTPUT_FILE"
    echo ""
    echo "First 20 lines of generated file:"
    echo "--------------------------------"
    head -20 "$OUTPUT_FILE"
else
    echo "❌ Error: File was not created"
    exit 1
fi
