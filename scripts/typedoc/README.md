# TypeDoc Documentation Generator for W&B Query Panel

## Overview

A TypeDoc-based documentation generator that creates beautiful, well-structured Hugo-compatible documentation from the W&B Query Expression Language TypeScript source code.

## ✨ Features

- **Rich Documentation** - Comprehensive docs with examples, parameters, and type information
- **Structure Examples** - Every data type shows complete TypeScript structure at the top
- **Hugo Compatible** - Outputs markdown with proper front matter and menu structure
- **Organized Structure** - Automatically groups operations and data types
- **Clean URLs** - No redundant prefixes in filenames
- **TSDoc Standard** - Uses industry-standard documentation comments

## 📁 Structure

Generated documentation is organized as:
```
query-panel/
├── _index.md              # Main entry
├── operations/            # Operation functions
│   ├── run-operations.md  # Lowercase kebab-case
│   └── artifact-operations.md
└── data-types/           # Type definitions
    ├── run.md            # All lowercase
    ├── artifact.md
    ├── configdict.md
    └── ...
```

## 🚀 Usage

### Generate Documentation

```bash
./generate-docs.sh /path/to/wandb/core
```

This will:
1. Read TypeScript source from `wandb/core/frontends/weave/src/core/ops/`
2. Generate markdown documentation using TypeDoc
3. Post-process for Hugo compatibility
4. Organize into proper structure
5. Add GitHub source links

### Output Location

Documentation is generated to:
```
/content/en/ref/query-panel/
```

## 🔧 Configuration

### TypeDoc Configuration (`typedoc.json`)
- Configured for markdown output with Hugo plugin
- Excludes test files and internal modules
- Groups content by category

### Post-Processing (`postprocess-hugo.js`)
- Adds Hugo front matter (title only, no H1s in content)
- Removes redundant prefixes from filenames
- Organizes files into logical directories
- Ensures all data types have structure examples
- Removes all Returns sections (redundant - type already in signature)
- Removes unnecessary bold Description headers
- Removes confusing "Chainable Operations Functions" header
- Promotes operations to H2 level for cleaner structure
- Keeps subsections (Parameters, Examples, See Also) at H4 to avoid TOC clutter
- Converts function signatures to proper code blocks (clean TypeScript, no markdown or escaping)
- Removes private repo information ("Defined in", "Since" sections)
- Removes TypeDoc's table of contents (Hugo auto-generates)
- Fixes broken references and corrects link paths
- Converts same-page anchor links to portable format (removes filename)
- Preserves specific cross-references between functions

## 📝 Features Added

### 1. Clean File Names
- Removes `W_B_Query_Expression_Language.` prefix
- Results in clean URLs like `/data-types/ConfigDict`

### 2. Nested Structure
- Operations grouped under `/operations/`
- Data types grouped under `/data-types/`
- Each section has cascade menus

### 3. Automatic Cleanup
- Removes redundant module files
- Deletes broken references
- Cleans up duplicate titles
- No GitHub Pages artifacts

## 📋 Prerequisites

- Node.js and npm
- Access to `wandb/core` repository
- TypeScript source files with TSDoc comments

## 🔄 Migration from Old System

The old system used a custom `generateDocs.ts` script that:
- Produced flat, unorganized documentation
- Lacked proper examples and type information
- Had no source links
- Generated poor formatting

This TypeDoc setup provides:
- Professional, organized documentation
- Rich examples and type information
- Direct GitHub source links
- Clean, maintainable structure

## 📚 Documentation Standards

Source code should use TSDoc comments:
```typescript
/**
 * Filter runs based on a condition
 * @param predicate - Function to test each run
 * @returns Filtered list of runs
 * @example
 * ```typescript
 * runs.filter(r => runSummary(r).accuracy > 0.9)
 * ```
 */
```

## 🎯 Benefits

1. **Better Developer Experience** - Clear, navigable documentation
2. **Maintainable** - Standard tools and clear structure
3. **Professional** - Consistent formatting and organization
4. **Automated** - Single command regenerates everything

## 📂 Files

- `generate-docs.sh` - Main generation script
- `postprocess-hugo.js` - Hugo-specific post-processing
- `typedoc.json` - TypeDoc configuration
- `tsconfig.json` - TypeScript configuration
- `package.json` - Dependencies

## 🚦 Status

✅ **Production Ready**

The documentation generator is fully functional and produces high-quality documentation suitable for production use.