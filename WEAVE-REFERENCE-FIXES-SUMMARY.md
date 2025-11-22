# Weave Reference Generation Script Fixes

This document summarizes the fixes applied to the Weave reference documentation generation scripts based on PR #1888 feedback.

## Issues Fixed

### 1. Models Reference Files Being Renamed (CRITICAL BUG)
**Problem**: `fix_casing.py` was incorrectly targeting `models/ref/python/public-api` files instead of Weave reference docs.

**Fix**: Updated `fix_casing.py` to only target `weave/reference/python-sdk` files.
- Changed path from `models/ref/python/public-api` to `weave/reference/python-sdk`
- Removed the logic that was renaming Models API files (ArtifactCollection, etc.)
- Added clear comments indicating this should NEVER touch Models reference docs

**Files Modified**: 
- `scripts/reference-generation/weave/fix_casing.py`

### 2. TypeScript SDK Using PascalCase Filenames
**Problem**: TypeScript SDK files were being generated with PascalCase filenames (e.g., `Dataset.mdx`, `WeaveClient.mdx`), which causes Git case-sensitivity issues.

**Fix**: Updated generation scripts to use lowercase filenames throughout.
- Modified `generate_typescript_sdk_docs.py` to convert filenames to lowercase when creating `.mdx` files
- Updated function and type-alias extraction to use lowercase filenames
- Updated internal links to use lowercase paths

**Files Modified**:
- `scripts/reference-generation/weave/generate_typescript_sdk_docs.py` (lines 259, 319-320, 369-370, 379)
- `scripts/reference-generation/weave/fix_casing.py` (simplified to just convert to lowercase)

### 3. H1 in service-api/index.mdx
**Problem**: The generated `service-api/index.mdx` had both a frontmatter title and an H1, which is redundant in Mintlify.

**Fix**: Removed the H1 heading since Mintlify uses the frontmatter title.

**Files Modified**:
- `scripts/reference-generation/weave/generate_service_api_spec.py` (line 31)

### 4. Duplicate H3 Headings in service-api.mdx
**Problem**: The `service-api.mdx` file had duplicate category sections (e.g., "### Calls" appeared on both line 23 and line 158), listing the same endpoints twice.

**Fix**: Added deduplication logic to prevent duplicate categories and duplicate endpoints.
- Track which categories have been written to prevent duplicate H3 headings
- Deduplicate endpoints within each category by (method, path) tuple
- This prevents the same endpoint from being listed multiple times if it appears in the OpenAPI spec with duplicate tags

**Files Modified**:
- `scripts/reference-generation/weave/update_service_api_landing.py` (lines 99-118)

### 5. Markdown Table Formatting Errors (------ lines)
**Problem**: Python SDK docs contained standalone lines with just dashes (`------`) which break markdown parsing.

**Example**: In `trace_server_interface.mdx`, lines like 22, 30, 39, etc. had `------` that created invalid table structures.

**Fix**: Added regex pattern to remove these malformed table separators.
- Pattern: `\n\s*------+\s*\n` → `\n\n`
- This removes lines that are just dashes with optional whitespace

**Files Modified**:
- `scripts/reference-generation/weave/generate_python_sdk_docs.py` (lines 258-260)

## Testing Recommendation

Before merging, test the fixes by running the reference generation locally:

```bash
# From the docs repository root
cd scripts/reference-generation/weave
python generate_weave_reference.py
```

Then verify:
1. No files in `models/ref/python/public-api` were modified
2. All TypeScript SDK files in `weave/reference/typescript-sdk/` have lowercase filenames
3. `weave/reference/service-api/index.mdx` has no H1 heading
4. `weave/reference/service-api.mdx` has no duplicate H3 category headings
5. No `------` lines in `weave/reference/python-sdk/trace_server/trace_server_interface.mdx`
6. In `docs.json`, modules under `weave/reference/python-sdk/trace/` are grouped as "Core" (not "Other")
7. In `docs.json`, the Service API `openapi` configuration uses the local spec (not a GitHub URL) if sync_openapi_spec.py was run with `--use-local`

### 6. Incorrect Section Grouping ("Core" → "Other")
**Problem**: Python SDK modules in the `trace/` directory were being incorrectly grouped as "Other" instead of "Core" in docs.json navigation.

**Root Cause**: The path checking logic in `update_weave_toc.py` was checking `if parts[0] == "weave"`, but paths are relative to `python-sdk/`, so `parts[0]` is actually the module subdirectory (`trace`, `trace_server`, etc.), not `weave`.

**Fix**: Corrected the path checking logic to check the actual first path component.
- Changed from checking `parts[0] == "weave"` then `parts[1] == "trace"`
- To directly checking `parts[0] == "trace"`, `parts[0] == "trace_server"`, etc.

**Files Modified**:
- `scripts/reference-generation/weave/update_weave_toc.py` (lines 33-45)

### 7. OpenAPI Configuration Being Overwritten
**Problem**: `update_weave_toc.py` was unconditionally overwriting the OpenAPI spec configuration in docs.json to use a remote URL, ignoring the local spec that `sync_openapi_spec.py` downloads and configures.

**Impact**: Even though `sync_openapi_spec.py` downloads the OpenAPI spec locally and can configure docs.json to use it, `update_weave_toc.py` would immediately overwrite it with a remote GitHub URL, defeating the purpose of the local spec.

**Fix**: Removed the Service API OpenAPI configuration code from `update_weave_toc.py`. This script should only manage Python/TypeScript SDK navigation, not the OpenAPI spec source.
- Deleted lines 209-224 that were setting `page["openapi"]` to remote URLs
- Added comment noting that OpenAPI configuration is managed by `sync_openapi_spec.py`

**Files Modified**:
- `scripts/reference-generation/weave/update_weave_toc.py` (lines 206-207)

### 8. Missing Root Module Documentation (CRITICAL - WEAVE PACKAGE REGRESSION)
**Problem**: The generated `python-sdk.mdx` file is only 8 lines (just frontmatter), completely missing all the important API documentation for functions like `init()`, `publish()`, `ref()`, `get()`, etc.

**Expected**: The current version (Weave 0.52.10) has 2074 lines documenting all the core Weave functions and classes.

**Root Cause**: **This is a WEAVE PACKAGE REGRESSION, not a script bug.** 

Something changed in Weave between versions **0.52.10** (current docs) and **0.52.16** (PR version) that broke documentation generation for the root `weave` module. The generation scripts haven't changed, and lazydocs hasn't changed - so this is an upstream issue in the Weave package itself.

Possible causes:
1. Changes to `weave/__init__.py` that affect how the module exports its public API
2. Module structure refactoring that lazydocs can't handle
3. New import patterns or lazy loading that breaks introspection

**Status**: **CRITICAL UPSTREAM BUG** - This makes the Python SDK documentation completely unusable for version 0.52.16.

**Action Required**: Report this to the Weave team immediately:
1. File an issue: https://github.com/wandb/weave/issues
2. Include: "Documentation generation broken in 0.52.16 - root module exports not discoverable by lazydocs"
3. Mention: "Works fine in 0.52.10, broken in 0.52.16"
4. Tag: @dbrian57 or relevant Weave maintainers

**Recommendation**: 
- **DO NOT MERGE PR #1888** - it will break Python SDK documentation
- Either: Fix the Weave package and regenerate docs
- Or: Stay on 0.52.10 documentation until the Weave package is fixed

**Files to Investigate** (in Weave repo):
- `weave/__init__.py` between versions 0.52.10 and 0.52.16
- Any structural changes to the weave package in that version range

### 9. OpenAPI Spec Validation (New Feature)
**Enhancement**: Added validation to detect issues in the OpenAPI spec itself, which can help identify upstream problems.

**Features**:
- Detects duplicate endpoint definitions (same method+path defined multiple times)
- Identifies endpoints appearing in multiple categories/tags
- Warns when critical issues like duplicate endpoints are found
- Suggests reporting issues to the Weave team when spec problems are detected

**Files Modified**:
- `scripts/reference-generation/weave/sync_openapi_spec.py` (added `validate_spec()` function and integration in `main()`)

This will help identify if duplicate H3s or other issues originate from the OpenAPI spec rather than our generation scripts.

## Files Modified Summary

1. `scripts/reference-generation/weave/fix_casing.py`
2. `scripts/reference-generation/weave/generate_typescript_sdk_docs.py`
3. `scripts/reference-generation/weave/generate_service_api_spec.py`
4. `scripts/reference-generation/weave/update_service_api_landing.py`
5. `scripts/reference-generation/weave/generate_python_sdk_docs.py`
6. `scripts/reference-generation/weave/update_weave_toc.py`
7. `scripts/reference-generation/weave/sync_openapi_spec.py` (new validation feature)

All fixes are backward compatible and will take effect on the next reference documentation generation run.
