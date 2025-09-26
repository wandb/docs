# Testing the Link Checker Action

This document explains how to test the link checker action locally before merging the PR.

## Quick Test

Run the provided test script:

```bash
./.github/test-link-checker.sh
```

This script will:
1. Create test content with various broken links
2. Generate a simulated lychee output
3. Run the fix_broken_links.mjs script
4. Show you the results and verify that generated docs are skipped

## Manual Testing

If you want to test with real links:

### 1. Install Dependencies

```bash
# Ensure you have Node.js installed (check .node-version for required version)
node --version
```

### 2. Create Test Content

Create a markdown file with some broken links:

```bash
cat > content/test-links.md <<'EOF'
# Test Links

- HTTP link: http://github.com/wandb/docs
- Missing slash: https://docs.wandb.ai/guides
- Broken link: https://example.com/404-not-found
EOF
```

### 3. Run Lychee Manually

```bash
# Install lychee if not available
# On macOS: brew install lychee
# On Linux: Check https://github.com/lycheeverse/lychee#installation

# Run lychee to generate JSON output
mkdir -p lychee
lychee --accept 200,429,403 \
       --user-agent 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)' \
       --scheme https --scheme http \
       --max-concurrency 5 \
       --max-retries 1 \
       --retry-wait-time 2 \
       --format json \
       --output ./lychee/out.json \
       'content/test-links.md'
```

### 4. Run the Fix Script

```bash
node .github/scripts/fix_broken_links.mjs
```

### 5. Check Results

- Look at the generated report: `cat .github/lychee-report.md`
- Check what files were modified: `git diff`
- Verify generated docs in `ref/js/` and `ref/python/` are NOT modified

## Testing in GitHub Actions

To test the full workflow in GitHub Actions:

1. Push your changes to the PR branch
2. The link checker runs on a schedule, but you can manually trigger it:
   - Go to Actions tab
   - Find "Link Checker" workflow
   - Click "Run workflow"
   - Select your branch

## What to Verify

1. **Auto-fixes are applied correctly:**
   - HTTP → HTTPS conversions
   - Trailing slash additions/removals
   - WWW additions/removals

2. **Generated docs are skipped:**
   - Files in `*/ref/js/*` should NOT be modified
   - Files in `*/ref/python/*` should NOT be modified

3. **Translated content IS processed:**
   - Files in `content/ja/` should be fixed
   - Files in `content/ko/` should be fixed

4. **Report is accurate:**
   - `.github/lychee-report.md` shows fixed links
   - Manual follow-ups are listed for truly broken links

5. **PR is created correctly:**
   - PR title and description are clear
   - Comments show manual follow-ups needed

## Cleanup

After testing:

```bash
# Remove test files
rm -rf content/test-broken-links content/test-links.md
rm -f lychee/out.json .github/lychee-report.md

# Reset any modified files
git checkout -- content/
```