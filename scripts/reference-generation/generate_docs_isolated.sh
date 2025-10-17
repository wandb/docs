#!/bin/bash
# Generate reference documentation in an isolated environment
# This avoids dependency conflicts and license issues in the main project

set -e

echo "=== Reference Documentation Generation (Isolated) ==="
echo "This script runs documentation generation in temporary virtual environments"
echo "to avoid dependency conflicts and license policy issues."
echo

# Create a temporary directory for our work
TEMP_DIR=$(mktemp -d)
echo "Created temporary directory: $TEMP_DIR"

# Cleanup function
cleanup() {
    echo "Cleaning up temporary directory..."
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# Generate Service API documentation (no Python dependencies needed)
echo
echo "=== Generating Service API Documentation ==="
python3 scripts/reference-generation/generate_service_api_spec.py

# Generate Python SDK documentation in isolated environment
echo
echo "=== Generating Python SDK Documentation ==="
(
    cd "$TEMP_DIR"
    python3 -m venv venv_python
    source venv_python/bin/activate
    pip install -q requests lazydocs
    python3 "$OLDPWD/scripts/reference-generation/generate_python_sdk_docs.py"
)

# Generate TypeScript SDK documentation in isolated environment
echo
echo "=== Generating TypeScript SDK Documentation ==="
(
    cd "$TEMP_DIR"
    python3 -m venv venv_typescript
    source venv_typescript/bin/activate
    pip install -q requests
    python3 "$OLDPWD/scripts/reference-generation/generate_typescript_sdk_docs.py"
)

echo
echo "=== Documentation generation complete! ==="
echo "All reference documentation has been generated successfully."
