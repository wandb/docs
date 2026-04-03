# Chained Redirect Detector

Detects chained redirects in the Mintlify `docs.json` configuration file.

## What are chained redirects?

A chained redirect occurs when a redirect destination is itself a redirect source, causing multiple hops to reach the final destination. For example:

```
/old-path → /intermediate-path → /new-path
```

When a user visits `/old-path`, they are redirected to `/intermediate-path`, which then redirects them to `/new-path`. This creates two HTTP redirects (two hops) instead of one.

## Why are chained redirects a problem?

1. **Performance**: Each redirect adds latency and requires an additional HTTP round trip.
2. **SEO**: Search engines may not follow long redirect chains.
3. **Maintenance**: Harder to track and update redirect rules.
4. **User experience**: Slower page loads, especially on slower connections.

## Usage

### Run locally

From the repository root:

```bash
python3 scripts/chained-redirect-detector/detect-chained-redirects.py
```

Or from the script directory:

```bash
cd scripts/chained-redirect-detector
python3 detect-chained-redirects.py
```

### Check a specific docs.json file

```bash
python3 scripts/chained-redirect-detector/detect-chained-redirects.py /path/to/docs.json
```

## Output

The script outputs:

1. Total number of redirects found
2. Number of chained redirects detected
3. Chains grouped by hop count (longest chains first)
4. Recommendations for fixing the chains

### Example output

```
Checking redirects in: /Users/you/docs/public-docs/docs.json

Found 150 total redirect(s)

⚠️  Found 3 chained redirect(s)!

Direct chains: 3

================================================================================
3 chain(s) with 2 hop(s):
================================================================================

1.
    → /platform/regions/us-east-01
       → /platform/regions/us-east/us-east-01
          → /platform/regions/general-access/us-east/us-east-01

2.
    → /products/storage/object-storage/concepts/policies
       → /products/storage/object-storage/concepts/policies/overview
          → /products/storage/object-storage/auth-access/policies
```

## Exit codes

- `0`: No chained redirects found
- `1`: Chained redirects detected

This allows the script to be used in CI/CD pipelines to fail builds when chained redirects are introduced.

## Fixing chained redirects

When chained redirects are found:

1. **Identify the final destination** - The last URL in the chain is where users should ultimately land.

2. **Update the initial redirect** - Change the initial source to redirect directly to the final destination.

3. **Remove intermediate redirects** - If the intermediate redirect is no longer used by other chains, remove it.

### Example fix

**Before:**

```json
{
  "redirects": [
    {
      "source": "/old-path",
      "destination": "/intermediate-path"
    },
    {
      "source": "/intermediate-path",
      "destination": "/new-path"
    }
  ]
}
```

**After:**

```json
{
  "redirects": [
    {
      "source": "/old-path",
      "destination": "/new-path"
    },
    {
      "source": "/intermediate-path",
      "destination": "/new-path"
    }
  ]
}
```

## GitHub Actions integration

To run this script as a GitHub Action, create a workflow file (for example, `.github/workflows/check-redirects.yaml`):

```yaml
name: Check for chained redirects

on:
  pull_request:
    paths:
      - 'public-docs/docs.json'
  push:
    branches:
      - main
    paths:
      - 'public-docs/docs.json'

jobs:
  check-redirects:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.x'

      - name: Check for chained redirects
        run: |
          python3 scripts/chained-redirect-detector/detect-chained-redirects.py
```

This workflow will:

- Run on PRs that modify `public-docs/docs.json`
- Run on pushes to `main` that modify `public-docs/docs.json`
- Fail the check if chained redirects are detected

## Requirements

- Python 3.6 or higher
- No external dependencies (uses only Python standard library)

## Limitations

- Only checks redirects within the same `docs.json` file
- Does not check for redirects defined in other systems (for example, Cloudflare, nginx)
- Does not validate that destination URLs actually exist as pages
