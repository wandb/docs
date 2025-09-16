# ✅ Private Repository Information Removed

## What Was Removed

Since `wandb/core` is a private repository, we've removed all references that could expose internal information:

### 1. ❌ GitHub Source Links
- Removed all `## Source` sections with CTA buttons
- Users can't access private repos anyway

### 2. ❌ "Defined in" Sections  
- Removed all file path references like:
  ```
  #### Defined in
  src/operations/artifact-operations.ts:41
  ```
- Total removed: 77 references across all files

### 3. ❌ "Since" Version Sections
- Removed all version information like:
  ```
  **`Since`**
  1.2.0
  ```
- Total removed: 29 version references
- Prevents disclosure of internal versioning

## Post-Processor Updates

The `postprocess-hugo.js` now automatically removes:
```javascript
// Remove "Defined in" sections - source repo is private
content = content.replace(/#### Defined in\n\n.+\n\n?/gm, '');

// Remove "Since" version sections - private repo version info
content = content.replace(/\*\*`Since`\*\*\n\n[\d.]+\n\n?/gm, '');
```

## Clean Documentation

### Before (with private info)
```markdown
**`Since`**

1.2.0

**`Example`**
...

#### Defined in

src/operations/artifact-operations.ts:41

## Source

{{< cta-button githubLink=https://github.com/wandb/core/... >}}
```

### After (clean)
```markdown
**`Example`**
...
```

## Benefits

- ✅ **No Private Info** - Internal paths and versions not exposed
- ✅ **No Broken Links** - No links to inaccessible private repos
- ✅ **Clean Output** - Documentation focuses on the API itself
- ✅ **Professional** - Public docs without implementation details

## What Remains

The documentation still includes:
- Complete API reference with all operations
- Comprehensive TypeScript examples
- Parameter descriptions and types
- Return value documentation
- Usage examples
- Cross-references between related functions

The documentation is now appropriate for public consumption without exposing any private repository details!
