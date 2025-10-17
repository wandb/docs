# Reference Documentation Generation Scripts

This directory contains scripts to automatically generate reference documentation for Weave from its source code, formatted for Mintlify.

## Overview

The system generates three types of documentation:

1. **Service API Documentation** - From the OpenAPI specification
2. **Python SDK Documentation** - From Python source code using lazydocs
3. **TypeScript SDK Documentation** - From TypeScript source code using typedoc

## Prerequisites

- Python 3.11+
- Node.js 18+
- npm

## Security Considerations

These scripts are intended for development/CI use only, not production environments:

1. **Python Dependencies**: We use minimal dependencies (`requests` and `lazydocs`) with pinned versions
2. **Network Access**: Scripts download from trusted sources (pypi.org, npm registry, trace.wandb.ai)
3. **File System**: Scripts write only to the local `weave/` directory structure
4. **No Sensitive Data**: Scripts don't handle authentication or sensitive information

### Addressing License Policy Violations

Socket Security may flag license policy violations for `lazydocs` and its dependencies. Since these are development-only tools used for documentation generation (not production code), they are excluded via `.socketignore`.

**Note**: `lazydocs` is maintained by W&B and is the preferred tool for generating Python SDK documentation. See [LICENSE_NOTICE.md](./LICENSE_NOTICE.md) for important information about dependency licenses.

If you still need alternatives:

1. **Use the isolated generation script**: Run `./generate_docs_isolated.sh` which creates temporary virtual environments for each documentation type, preventing dependency conflicts in the main project

2. **Use the minimal Python generator**: Run `./generate_python_sdk_minimal.py` as a fallback option if lazydocs cannot be used

3. **Run in CI/Docker**: Generate documentation in a containerized environment where license policies don't affect the main repository

## Setup

1. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install Python dependencies:
   ```bash
   pip install -r scripts/reference-generation/requirements.txt
   ```

## Usage

### Automated Generation via GitHub Actions (Recommended)

The primary way to generate reference documentation is through the GitHub Actions workflow:

1. Go to the [Actions tab](https://github.com/wandb/mintlifytest/actions) in the repository
2. Select "Generate Reference Documentation" workflow
3. Click "Run workflow"
4. Enter the Weave version:
   - `latest` - Latest PyPI release (default)
   - `0.51.34` or `v0.51.34` - Specific version
   - Commit SHA - Specific commit
   - Branch name - From a branch
5. Click "Run workflow"

The workflow will:
- Generate all three types of documentation
- Fix any broken internal links
- Create a draft PR with the changes for review

### Manual Local Generation

For development or testing, you can run the generators locally:

```bash
# Option 1: Use the test script (runs all generators in isolated environments)
./scripts/reference-generation/test-locally.sh latest

# Option 2: Run individual generators
# Generate Service API docs
python scripts/reference-generation/generate_service_api_spec.py

# Generate Python SDK docs (specify version as argument)
python scripts/reference-generation/generate_python_sdk_docs.py latest

# Generate TypeScript SDK docs (specify version as argument)
python scripts/reference-generation/generate_typescript_sdk_docs.py latest

# Fix broken links after generation
python scripts/reference-generation/fix_broken_links.py
```

## Output Structure

The scripts generate documentation in the following structure:

```
weave/
├── api-reference/
│   └── openapi.json          # Service API OpenAPI spec
└── reference/
    ├── python-sdk/
    │   └── weave/
    │       ├── index.mdx     # Main module docs
    │       ├── trace/
    │       │   ├── op.mdx
    │       │   ├── weave_client.mdx
    │       │   └── util.mdx
    │       └── trace_server/
    │           └── trace_server_interface.mdx
    └── typescript-sdk/
        └── weave/
            ├── index.mdx
            ├── classes/
            ├── functions/
            ├── interfaces/
            └── type-aliases/
```

## Testing Locally

After generating documentation, you can test it with Mintlify:

```bash
# From the project root
mint dev
```

Then navigate to the reference documentation sections to verify the output.

## How It Works

### Service API (`generate_service_api_spec.py`)
- Downloads the OpenAPI spec from https://trace.wandb.ai/openapi.json
- Updates server URLs and metadata
- Saves directly for Mintlify to consume (no conversion needed)

### Python SDK (`generate_python_sdk_docs.py`)
- Installs the specified Weave version
- Uses lazydocs to generate markdown documentation
- Converts from Docusaurus format to Mintlify MDX format
- Handles special cases like Pydantic models
- Fixes code fence indentation issues

### TypeScript SDK (`generate_typescript_sdk_docs.py`)
- Downloads Weave source code for the specified version
- Installs dependencies and typedoc
- Generates markdown documentation
- Converts to Mintlify MDX format
- Organizes files according to Mintlify structure

## Verifying Generated Documentation

After generation (either via GitHub Actions or locally):

1. **Check Generated Files**:
   - `weave/api-reference/openapi.json` - Service API spec
   - `weave/reference/python-sdk/` - Python SDK docs
   - `weave/reference/typescript-sdk/` - TypeScript SDK docs

2. **Validate Links**:
   ```bash
   mintlify broken-links
   ```
   All internal links should work correctly.

3. **Review PR** (if generated via GitHub Actions):
   - A draft PR titled "[Draft] Update reference documentation (Weave X.X.X)"
   - Labels: `documentation`, `automated`, `weave`
   - Description includes version info and changes made

## Troubleshooting

### GitHub Actions Issues
- **Permission errors**: The workflow needs write permissions to create PRs
- **Package not found**: Check the Weave version exists on PyPI
- **TypeScript download fails**: Check the npm registry for the @wandb/weave package

### Python SDK Issues
- If module imports fail, check that Weave installed correctly
- For development versions, ensure you have git access to the Weave repository

### TypeScript SDK Issues
- Ensure Node.js 18+ is installed
- If typedoc fails, check for TypeScript compilation errors in the Weave SDK
- The script pins specific versions of typedoc that are known to work

### General Issues
- Check that you're in the activated virtual environment
- Ensure you have internet access for downloading packages and source code
- For version-specific issues, try using `latest` or `main` as a fallback
- If broken links persist after generation, run `fix_broken_links.py` manually