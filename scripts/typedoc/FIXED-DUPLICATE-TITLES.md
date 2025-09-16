# ✅ Fixed Duplicate Title Issue

## The Problem
TypeDoc was generating files with duplicate title entries:
```markdown
---
title: ConfigDict
---
title: ConfigDict    <-- This was appearing in the content!
```

## The Solution
Updated the post-processing script (`postprocess-hugo.js`) to:

1. **Remove duplicate titles on line 4** - Common location for this issue
2. **Clean up excessive blank lines** after front matter
3. **Prevent future occurrences** when regenerating docs

## What Changed

### In `/scripts/typedoc/postprocess-hugo.js`:
```javascript
// Added logic to remove duplicate titles
const lines = content.split('\n');
if (lines[3] && lines[3].startsWith('title:')) {
  lines[3] = '';
}

// Also removes excessive blank lines
content = content.replace(/^(---\n[\s\S]*?\n---\n)\n+/, '$1\n');
```

## Result
✅ All existing files have been cleaned
✅ Future generations won't have this issue
✅ Front matter is properly formatted

## Files Fixed
- All files in `/content/en/ref/query-panel-new/`
- All files in `/content/en/ref/query-panel-new/interfaces/`
- All files in `/content/en/ref/query-panel-new/modules/`

Now when you run the generator script, it won't create duplicate titles!
