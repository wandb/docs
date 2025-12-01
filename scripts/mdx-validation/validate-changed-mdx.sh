#!/bin/bash
set -e

# This script validates only the MDX files that were changed in the PR
# It's much faster than running mint dev on all files

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 VALIDATING CHANGED MDX FILES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Get list of changed MDX files
CHANGED_FILES=$(git diff --name-only origin/$GITHUB_BASE_REF...HEAD -- '*.mdx')

if [ -z "$CHANGED_FILES" ]; then
  echo "No MDX files were changed"
  exit 0
fi

# Count changed files
FILE_COUNT=$(echo "$CHANGED_FILES" | wc -l)
echo "Found $FILE_COUNT changed MDX file(s):"
echo "$CHANGED_FILES" | sed 's/^/  - /'
echo ""

# If more than 20 files changed, fall back to full validation
# as it might be a large refactor
if [ "$FILE_COUNT" -gt 20 ]; then
  echo "More than 20 MDX files changed, running full validation..."
  exec ./scripts/mdx-validation/validate-mdx-mintlify.sh
fi

# Create a temporary directory for validation
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Copy only necessary files for validation
echo "Setting up validation environment..."
cp docs.json "$TEMP_DIR/"

# Copy changed MDX files while preserving directory structure
for file in $CHANGED_FILES; do
  dir=$(dirname "$file")
  mkdir -p "$TEMP_DIR/$dir"
  cp "$file" "$TEMP_DIR/$file"
done

# Also copy any referenced snippets or includes
# This is a simplified approach - may need refinement based on your setup
if [ -d "snippets" ]; then
  cp -r snippets "$TEMP_DIR/"
fi

# Run Mintlify validation in the temp directory
echo ""
echo "Running Mintlify validation on changed files..."
echo ""

cd "$TEMP_DIR"
LOGFILE="/tmp/mint-validate-$$.log"

# Run mint dev briefly to check for parsing errors
# Use shorter timeout since we're only checking specific files
timeout --preserve-status 15s mint dev --no-open 2>&1 | tee "$LOGFILE" > /dev/null || true

echo ""
echo "✓ Mintlify finished parsing"
echo ""

# Check for parsing errors
if grep -q "parsing error" "$LOGFILE"; then
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "❌ MINTLIFY PARSING ERRORS DETECTED"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""
  echo "Parsing errors found:"
  echo ""
  grep "parsing error" "$LOGFILE" | sed 's/^/  /'
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "💡 These are Mintlify parsing errors. Please fix them or"
  echo "   file an issue if you believe they are incorrect:"
  echo "   https://github.com/wandb/docs/issues/new"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo ""
  rm -f "$LOGFILE"
  exit 1
else
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "✅ MINTLIFY VALIDATION PASSED"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "All changed MDX files validated successfully"
  rm -f "$LOGFILE"
  exit 0
fi
