## Summary

This PR fixes redirect issues that were causing certain pages to return 404 errors instead of redirecting properly, despite having entries in the Cloudflare `_redirects` file.

## Problem

1. **Cloudflare Pages Limitation**: URLs without trailing slashes don't match redirect rules that include trailing slashes. For example, `/app/features` doesn't match the redirect rule `/app/features/ /guides/app/ 301`.

2. **Edge Function Removed**: A Cloudflare edge function that was handling trailing slashes was removed due to rate limiting issues (commit 4396d5f9), and even when it existed, it didn't include `/app/` paths in its coverage.

3. **Conflicting Aliases**: Japanese and Korean translation files still contained Hugo aliases that were creating client-side redirects, potentially conflicting with Cloudflare server-side redirects.

## Solution

### 1. Removed remaining Hugo aliases from translation files:
- `content/ja/guides/models/app/_index.md`
- `content/ko/guides/models/app/_index.md`

These files contained:
```yaml
aliases:
- /guides/models/app/features
```

### 2. Added redirect rules without trailing slashes:
Added duplicate redirect rules in `static/_redirects` to handle both URL formats:
```
/app/features /guides/app/ 301
/app/features/ /guides/app/ 301
/guides/models/app/features /guides/app/ 301
/guides/models/app/features/ /guides/app/ 301
```

## Testing

⚠️ **Important**: These changes need to be verified using the Cloudflare preview build, as redirects are handled at the CDN level, not by Hugo.

### To Test:
1. Once the preview build is ready, test these URLs:
   - `/app/features` (without trailing slash)
   - `/app/features/` (with trailing slash)
   - `/guides/models/app/features` (without trailing slash)
   - `/guides/models/app/features/` (with trailing slash)

2. All should redirect to `/guides/app/`

## Related

- Completes cleanup started in PR #1642 (commit 9b950cd8)
- Addresses issues after edge function removal (commit 4396d5f9)