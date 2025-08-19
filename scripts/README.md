# Link Checking Scripts

This directory contains scripts for processing and fixing broken links in the documentation.

## Scripts

### process_lychee_output.py

Processes the raw JSON output from lychee link checker into a human-readable markdown report.

**Usage:**
```bash
python3 scripts/process_lychee_output.py <input.json> <output.md>
```

**Features:**
- Categorizes links by status (broken, redirected, etc.)
- Groups errors by status code for easier review
- Provides actionable summaries and recommendations
- Generates clean markdown reports suitable for GitHub issues/PRs

### auto_fix_links.py

Automatically fixes common link issues based on lychee output.

**Usage:**
```bash
python3 scripts/auto_fix_links.py <lychee_output.json>
```

**Auto-fixes:**
- Updates redirected links to their final destinations
- Converts http:// to https:// where appropriate
- Removes unnecessary trailing slashes
- Fixes GitHub raw content URLs
- Generates a summary of all fixes applied

## GitHub Actions Integration

These scripts are integrated into two workflows:

1. **linkcheck.yml**: Monthly link check that creates GitHub issues with human-readable reports
2. **linkcheck-pr.yml**: Creates PRs with auto-fixed links when possible

## Example Output

The processed report includes:
- Summary statistics
- Broken links grouped by error type
- Redirected links with their destinations
- Clear action items for manual fixes