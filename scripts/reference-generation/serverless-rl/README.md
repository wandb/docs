# Training API Reference Documentation Scripts

This directory contains scripts to manage and update the Training API reference documentation for W&B Training.

## Overview

The Training API documentation system manages:
1. **OpenAPI Specification** - Downloads and maintains a local copy of the spec
2. **Landing Page** - Updates the api-reference.mdx page with current endpoints

## Scripts

### `sync_openapi_spec.py`
Downloads the latest OpenAPI spec from the Training API service and saves it locally.

```bash
# Sync the spec (downloads if changed)
python scripts/reference-generation/training/sync_openapi_spec.py

# Configure to use local spec
python scripts/reference-generation/training/sync_openapi_spec.py --use-local

# Configure to use remote spec  
python scripts/reference-generation/training/sync_openapi_spec.py --use-remote
```

### `update_training_api_landing.py`
Updates the Training API landing page with the current list of endpoints from the OpenAPI spec.

```bash
python scripts/reference-generation/training/update_training_api_landing.py
```

### `generate_training_reference.py`
Master script that runs all Training API generation steps in order.

```bash
python scripts/reference-generation/training/generate_training_reference.py
```

## GitHub Actions Workflow

The `update-training-api.yml` workflow:
- Runs weekly on Mondays at 9:30 AM UTC
- Can be triggered manually via workflow_dispatch
- Only creates a PR if the API has actually changed

## Output Structure

```
training/
├── api-reference.mdx         # Landing page with endpoints list
└── api-reference/
    └── openapi.json          # Local copy of OpenAPI spec
```

## Configuration

The system uses the local OpenAPI spec by default (configured in `docs.json`):
```json
{
  "group": "API Reference",
  "openapi": "training/api-reference/openapi.json",
  "pages": ["training/api-reference"]
}
```

This provides:
- **Reliable builds** - No dependency on external service availability
- **Version control** - Track API changes in git history
- **Faster builds** - No network fetch required during documentation builds

## Why Local Spec?

We use a local copy of the OpenAPI spec to avoid 502 Bad Gateway errors when running `mint dev` locally. The spec is automatically synced weekly via GitHub Actions, ensuring it stays up to date while maintaining build reliability.

## Testing

After running the scripts:

1. **Check the generated files:**
   ```bash
   ls -la training/api-reference/openapi.json
   cat training/api-reference.mdx | grep "## Available Endpoints" -A 20
   ```

2. **Test locally with Mintlify:**
   ```bash
   mint dev
   ```
   Navigate to `/training/api-reference` to verify the endpoints are listed correctly.

3. **Validate configuration:**
   ```bash
   grep "training/api-reference" docs.json
   ```

## Known Issues and Workarounds

### Missing Tags on Health Endpoints (RESOLVED)

**Issue**: The upstream Training API's OpenAPI spec was missing `tags` fields on the health endpoints (`/v1/health` and `/v1/system-check`). This caused Mintlify to place these endpoints in an incorrect navigation hierarchy with an extra "API Reference" layer in the breadcrumb.

**Status**: ✅ **FIXED** - The upstream fix was deployed on January 21, 2026 via [PR #213](https://github.com/coreweave/serverless-training/pull/213). The health endpoints now correctly include `"tags": ["health"]` in the OpenAPI spec.

**Temporary Workaround**: The `sync_openapi_spec.py` script includes a `patch_spec()` function that automatically adds the missing tags when syncing. This workaround can be removed once the next sync pulls the fixed spec from production.

## Troubleshooting

### 502 Bad Gateway Errors
If you're getting 502 errors with `mint dev`, ensure you're using the local spec:
```bash
python scripts/reference-generation/training/sync_openapi_spec.py --use-local
```

### Missing Endpoints
If endpoints aren't showing up:
1. Sync the latest spec: `python scripts/reference-generation/training/sync_openapi_spec.py`
2. Update the landing page: `python scripts/reference-generation/training/update_training_api_landing.py`

### Remote Spec Unavailable
If the remote spec can't be fetched, the scripts will fall back to using the existing local copy if available.
