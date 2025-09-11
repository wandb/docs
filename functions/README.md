# Cloudflare Pages Functions

This directory contains Cloudflare Pages Functions that enhance the functionality of the docs site.

## Middleware

### `_middleware.js` - Trailing Slash Redirect Handler

This middleware addresses a known Cloudflare Pages issue where URLs without trailing slashes don't match redirect rules in the `_redirects` file.

**Problem it solves:**
- `/wandb/config` → 404 (doesn't match redirect rule)
- `/wandb/config/` → redirects correctly

**How it works:**
1. Intercepts requests to URLs without trailing slashes
2. Checks if adding a slash would match a redirect rule
3. If yes, redirects to the URL with a trailing slash
4. The browser then follows the actual redirect rule

**Example:**
- User visits: `/wandb/config`
- Middleware redirects to: `/wandb/config/`
- _redirects rule redirects to: `/ref/python/sdk/functions/init/`

This is a temporary solution until the migration off Cloudflare Pages is completed.
