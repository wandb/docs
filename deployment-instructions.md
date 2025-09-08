# Preview Workflow Fix Deployment Instructions

## Current Situation

I've created fixes for the HTML preview workflow issues and prepared two versions:

### 1. Testing Version (with temporary override)
- **File**: `testing-version-with-override.patch`
- **Branch**: `testing-with-override`
- **Purpose**: For testing in your fork's main branch
- **Contains**: All fixes + temporary hardcoded Cloudflare preview URL from PR #1607

### 2. Production Version (clean)
- **File**: `production-version-clean.patch`
- **Branch**: `fix-preview-workflow-issues`
- **Purpose**: For the final PR to wandb/docs
- **Contains**: Only the fixes, no temporary overrides

## Deployment Steps

### Step 1: Test in Your Fork

1. Apply the testing version to your fork's main branch:
   ```bash
   # In your local clone of mdlinville/docs
   git checkout main
   git apply testing-version-with-override.patch
   git add .
   git commit -m "Test preview workflow fixes with temporary override"
   git push origin main
   ```

2. Create a test PR in your fork with various file changes:
   - Add a new file
   - Modify an existing file (including _index.md)
   - Delete a file
   - Modify an include file
   - Rename/move a file

3. Verify the preview comment shows:
   - Added files ARE linked
   - Modified files ARE linked (including index files)
   - Deleted files are NOT linked
   - Include files show dependent pages with links

### Step 2: Deploy to Production

Once testing is successful:

1. Apply the clean version to a new branch:
   ```bash
   # In your local clone of wandb/docs
   git checkout main
   git pull origin main
   git checkout -b fix-preview-workflow-issues
   git apply production-version-clean.patch
   git add .
   git commit -m "Fix preview workflow issues

   - Fix added files not being linked by checking out PR branch before Hugo build
   - Fix include file path construction when searching for dependent pages
   - Add special handling for index files in path mapping
   - Prevent deleted files from having preview links
   - Pass isDeleted flag to buildRows function to control link generation"
   git push origin fix-preview-workflow-issues
   ```

2. Create a PR from `fix-preview-workflow-issues` to `wandb/docs:main`

## What Was Fixed

1. **Added files not linked**: Now checks out PR branch before Hugo build
2. **Includes without links**: Fixed path construction for finding dependent pages
3. **Modified index not linked**: Added special handling for _index.md files
4. **Deleted files with links**: Added isDeleted flag to prevent link generation

## Testing Override Details

The testing version uses a hardcoded URL: `https://1607.docs-14y.pages.dev`

This override is in two places:
- `.github/workflows/pr-preview-links.yml` (line ~224)
- `.github/workflows/pr-preview-links-on-comment.yml` (line ~131)

Look for comments marked "TEMPORARY OVERRIDE FOR TESTING - Remove before production"