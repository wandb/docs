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
   pip install -r scripts/reference-generation/weave/requirements.txt
   ```

## Usage

### Automated Generation via GitHub Actions (Recommended)

The primary way to generate reference documentation is through the GitHub Actions workflows:

#### Full Reference Generation
1. Go to the [Actions tab](https://github.com/wandb/mintlifytest/actions) in the repository
2. Select "Generate Weave references" workflow
3. Click "Run workflow"
4. Enter the Weave version:
   - `latest` - Latest PyPI release (default)
   - `0.51.34` or `v0.51.34` - Specific version
   - Commit SHA - Specific commit
   - Branch name - From a branch
5. Click "Run workflow"

The workflow will:
- Sync the OpenAPI specification from the remote service
- Update the Service API landing page with current endpoints
- Generate Python and TypeScript SDK documentation
- Fix any broken internal links
- Convert Service API links to fully qualified URLs
- Create a draft PR with the changes for review

#### Service API Updates Only
There's also a dedicated workflow for Service API updates:
- **Manual trigger**: Run "Update Service API Documentation" workflow
- **Automatic**: Runs weekly (Mondays at 9 AM UTC)
- Only creates a PR if the API has actually changed

### Manual Local Generation

For development or testing, you can run the generators locally:

```bash
# Option 1: Use the master generation script (recommended)
python scripts/reference-generation/weave/generate_weave_reference.py

# Option 2: Use the test script (runs all generators in isolated environments)
./scripts/reference-generation/weave/test-locally.sh latest

# Option 3: Run individual generators
# Sync OpenAPI spec from remote service
python scripts/reference-generation/weave/sync_openapi_spec.py

# Generate Service API docs
python scripts/reference-generation/weave/generate_service_api_spec.py

# Update Service API landing page with current endpoints
python scripts/reference-generation/weave/update_service_api_landing.py

# Generate Python SDK docs (specify version as argument)
python scripts/reference-generation/weave/generate_python_sdk_docs.py latest

# Generate TypeScript SDK docs (specify version as argument)
python scripts/reference-generation/weave/generate_typescript_sdk_docs.py latest

# Fix broken links after generation
python scripts/reference-generation/weave/fix_broken_links.py

# Convert Service API links to fully qualified URLs
python scripts/reference-generation/weave/fix_service_api_links.py

# Update TOC in docs.json
python scripts/reference-generation/weave/update_weave_toc.py
```

## Output Structure

The scripts generate documentation in the following structure:

```
weave/
└── reference/
    ├── service-api/
    │   ├── service-api.mdx   # Landing page with endpoints list
    │   └── openapi.json      # Service API OpenAPI spec (local copy)
    ├── python-sdk/
    │   ├── python-sdk.mdx    # Main module docs
    │   ├── trace/
    │   │   ├── op.mdx
    │   │   ├── weave_client.mdx
    │   │   └── util.mdx
    │   └── trace_server/
    │       └── trace_server_interface.mdx
    └── typescript-sdk/
        ├── typescript-sdk.mdx # Main SDK docs
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

### Service API
The Service API documentation uses a multi-step process:

1. **`sync_openapi_spec.py`**:
   - Downloads the latest OpenAPI spec from https://trace.wandb.ai/openapi.json
   - Compares with local copy to detect changes
   - Saves to `weave/reference/service-api/openapi.json`
   - Can switch between local and remote spec in docs.json
   
   **Note**: The documentation uses the local OpenAPI spec by default for better build reliability. The spec is automatically synced weekly via GitHub Actions.

2. **`generate_service_api_spec.py`**:
   - Sets up the Service API documentation structure
   - Creates the service-api directory if needed

3. **`update_service_api_landing.py`**:
   - Parses the OpenAPI spec to extract all endpoints
   - Groups endpoints by category (Calls, Costs, Feedback, etc.)
   - Updates the service-api.mdx landing page with current endpoints list
   - Generates fully qualified URLs for each endpoint

4. **`fix_service_api_links.py`**:
   - Scans all documentation for Service API references
   - Converts internal links to fully qualified URLs (https://docs.wandb.ai/...)
   - Prevents broken link warnings for dynamically generated pages

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

### Master Generation Script (`generate_weave_reference.py`)
This script orchestrates all the generation steps in the correct order:
1. Syncs the OpenAPI specification
2. Generates Python SDK documentation
3. Generates TypeScript SDK documentation
4. Updates Service API landing page
5. Fixes broken links (general)
6. Fixes Service API links (converts to fully qualified URLs)
7. Updates the TOC in docs.json

This is the recommended way to run all generators locally.

## Verifying Generated Documentation

After generation (either via GitHub Actions or locally):

1. **Check Generated Files**:
   - `weave/reference/service-api/openapi.json` - Local copy of Service API spec
   - `weave/reference/service-api.mdx` - Service API landing page with endpoints
   - `weave/reference/python-sdk/` - Python SDK docs
   - `weave/reference/typescript-sdk/` - TypeScript SDK docs

2. **Validate Links**:
   ```bash
   mintlify broken-links
   ```
   All internal links should work correctly. Service API links will be fully qualified URLs.

3. **Review PR** (if generated via GitHub Actions):
   - A draft PR titled "Update Weave reference documentation (Weave X.X.X)"
   - Labels: `documentation`, `automated`, `weave`
   - Description includes:
     - Version info and timestamp
     - List of changes (OpenAPI sync, SDK updates, link fixes)
     - Link to source code for the version

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