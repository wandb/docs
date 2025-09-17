# Cloudflare Preview Build Test Plan

## Prerequisites
1. Wait for Cloudflare preview build to complete on the PR
2. Get the preview URL from the PR comments

## Test Cases

### 1. Primary Redirect Tests
Test these URLs on the preview build:

| Test URL | Expected Redirect | Notes |
|----------|-------------------|-------|
| `/app/features` | `/guides/app/` | Without trailing slash - main fix |
| `/app/features/` | `/guides/app/` | With trailing slash - should already work |
| `/guides/models/app/features` | `/guides/app/` | Without trailing slash |
| `/guides/models/app/features/` | `/guides/app/` | With trailing slash |

### 2. Verify No Circular Redirects
- Ensure none of the above URLs result in redirect loops
- Check browser network tab to confirm single 301 redirect

### 3. Language-specific Pages
Test that translation pages don't have conflicting redirects:
- `/ja/guides/models/app/` - Should load normally (no redirect loop)
- `/ko/guides/models/app/` - Should load normally (no redirect loop)

### 4. Related Redirects Still Work
Spot check a few other app-related redirects:
- `/app/` → `/guides/app/`
- `/app/features/panels/` → `/guides/app/features/panels/`

## How to Test
1. Open browser developer tools (Network tab)
2. Navigate to each test URL
3. Verify:
   - 301 redirect status code
   - Correct destination URL
   - No redirect chains or loops
   - Final page loads successfully

## Success Criteria
- All test URLs redirect to their expected destinations
- No 404 errors
- No redirect loops
- No console errors about too many redirects