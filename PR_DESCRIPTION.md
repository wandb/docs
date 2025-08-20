## Problem

In PR #1543, the HTML preview GitHub Action was showing the old page title "Query API" instead of the new title "Public API" that was updated in the PR. This happened because:

1. The PR had malformed YAML front matter (missing the closing `---` delimiter)
2. The GitHub Action workflows were relying solely on Hugo's generated `pageurls.json` file for titles
3. When Hugo couldn't parse the front matter correctly, it would fall back to default titles

## Solution

This PR enhances both preview link workflows (`pr-preview-links.yml` and `pr-preview-links-on-comment.yml`) to:

1. **Add a fallback mechanism** that reads titles directly from markdown files' front matter
2. **Maintain backward compatibility** by trying these sources in order:
   - First: Title from `pageurls.json` (Hugo's generated metadata)
   - Second: Title extracted directly from the markdown file's front matter
   - Third: Title generated from the file path (existing fallback)

## Implementation Details

Added an `extractTitleFromMarkdown()` function that:
- Reads the markdown file content
- Extracts YAML front matter using regex
- Parses the title field
- Handles quoted and unquoted title values
- Gracefully fails if the file can't be read

This ensures that PR preview links will always show the most up-to-date title, even if:
- The PR modifies the title in the markdown front matter
- The front matter is temporarily malformed during editing
- Hugo fails to properly parse the title for any reason

## Testing

The fix will be validated when this PR's preview links are generated, showing that titles are correctly extracted from the modified files.