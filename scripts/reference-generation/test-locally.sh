#!/bin/bash
# Test reference documentation generation locally
# Usage: ./test-locally.sh [version]
# Example: ./test-locally.sh 0.51.34

set -e

VERSION=${1:-latest}
echo "Testing reference documentation generation with Weave version: $VERSION"

# Create a temporary virtual environment
echo "Creating temporary environment..."
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

# Test Service API generation
echo ""
echo "=== Testing Service API Generation ==="
python3 -m venv venv_api
source venv_api/bin/activate
pip install -q requests
python3 "$OLDPWD/scripts/reference-generation/generate_service_api_spec.py"
deactivate
rm -rf venv_api

# Test Python SDK generation
echo ""
echo "=== Testing Python SDK Generation ==="
python3 -m venv venv_python
source venv_python/bin/activate
pip install -q requests lazydocs
python3 "$OLDPWD/scripts/reference-generation/generate_python_sdk_docs.py" "$VERSION"
deactivate
rm -rf venv_python

# Test TypeScript SDK generation
echo ""
echo "=== Testing TypeScript SDK Generation ==="
python3 -m venv venv_typescript
source venv_typescript/bin/activate
pip install -q requests
python3 "$OLDPWD/scripts/reference-generation/generate_typescript_sdk_docs.py" "$VERSION"
deactivate
rm -rf venv_typescript

# Go back to original directory
cd "$OLDPWD"

# Fix broken links
echo ""
echo "=== Fixing Broken Links ==="
python3 scripts/reference-generation/fix_broken_links.py

# Clean up
rm -rf "$TEMP_DIR"

echo ""
echo "✅ All tests completed successfully!"
echo "Generated documentation for Weave version: $VERSION"
echo ""
echo "You can now:"
echo "1. Run 'mint broken-links' to check for broken links"
echo "2. Run 'mint dev' to preview the documentation"
echo "3. Commit and push the changes if everything looks good"
