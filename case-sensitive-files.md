# Case-Sensitive Files Documentation

## Overview

This document tracks files in the repository that use uppercase letters in their names. On macOS (case-insensitive filesystem), special care is needed when working with these files.

## Current Status

- **Total files with uppercase**: 40
- **Case conflicts**: 0 (no files that differ only in case)
- **Git case sensitivity**: Configured with `core.ignorecase = false`

## Files Requiring Uppercase (Intentional)

### Python SDK Public API Reference (14 files)
Location: `models/ref/python/public-api/`

These files use uppercase naming to match Python class names:
- `ArtifactCollection.mdx`
- `ArtifactCollections.mdx`
- `ArtifactFiles.mdx`
- `ArtifactType.mdx`
- `ArtifactTypes.mdx`
- `BetaReport.mdx`
- `Member.mdx`
- `Project.mdx`
- `Registry.mdx`
- `Run.mdx`
- `RunArtifacts.mdx`
- `Sweep.mdx`
- `Team.mdx`
- `User.mdx`

**Note**: These were restored from the Lazydocs update (Sept 17, 2025) and must maintain uppercase naming to match the Python SDK conventions.

### Weave TypeScript SDK Reference
Location: `weave/reference/typescript-sdk/weave/`

Auto-generated from TypeScript definitions (likely need to keep uppercase):
- Classes: `WeaveClient.mdx`, `StringPrompt.mdx`, `Dataset.mdx`, etc.
- Interfaces: `WeaveAudio.mdx`, `WeaveImage.mdx`, `CallSchema.mdx`, etc.
- Functions: Various mixed-case function names
- Type aliases: `Op.mdx`

### Weave Cookbooks (3 files)
Location: `weave/cookbooks/`
- `Intro_to_Weave_Hello_Trace.mdx`
- `Models_and_Weave_Integration_Demo.mdx`
- `Intro_to_Weave_Hello_Eval.mdx`

These appear to be notebook-style tutorials with descriptive names.

## Handling Case-Sensitive Files on macOS

### Best Practices

1. **Always use `git config core.ignorecase false`** in this repository
2. **Be careful with `git mv`** - it may not properly handle case changes
3. **To rename case only**: 
   ```bash
   git mv oldname.mdx tempname.mdx
   git mv tempname.mdx NewName.mdx
   ```
4. **Check both git and filesystem** when verifying uppercase files exist

### Known Issues

- macOS filesystem is case-insensitive by default
- Git on macOS may show confusing status for case-only changes
- Some tools may not distinguish between `file.mdx` and `File.mdx`

### Verification Commands

```bash
# Check if git tracks uppercase files correctly
git ls-files | grep -E "[A-Z]" | wc -l

# Check actual files on disk
find . -name "*[A-Z]*" -type f | wc -l

# Ensure git is case-sensitive
git config core.ignorecase
```

## Recommendations

1. **Python SDK files**: Must keep uppercase naming (matches Python classes)
2. **TypeScript SDK files**: Likely auto-generated, should keep as-is
3. **Weave cookbooks**: Could be renamed to lowercase for consistency, but not critical
4. **Future files**: Default to lowercase unless there's a specific reason for uppercase

## Migration Notes

During the Mintlify migration, the Python SDK reference files were incorrectly converted to lowercase. This has been fixed in commit 489bad24a. The uppercase naming is intentional and matches the Python SDK's class naming conventions.
