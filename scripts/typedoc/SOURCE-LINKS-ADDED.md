# ✅ GitHub Source Links Added

## Feature
Added automatic GitHub source links to all generated documentation pages.

## What It Does

Each documentation page now includes:
- A **Source** section with a direct link to the GitHub repository
- Links point to the actual TypeScript source files in `wandb/core`
- For interfaces, links include anchors (e.g., `types.ts#ConfigDict`)

## Example

In `ConfigDict.md`:
```markdown
## Source

View the source code on GitHub: [ConfigDict](https://github.com/wandb/core/blob/master/frontends/weave/src/core/ops/types.ts#ConfigDict)

## See Also

- [Query Panels Guide](/guides/models/app/features/panels/query-panels/)
- [W&B API Reference](/ref/python/)
```

## File Mapping

The script intelligently maps documentation files to source files:

| Documentation File | GitHub Source |
|-------------------|---------------|
| `Run_Operations.md` | `runOperations.ts` |
| `Artifact_Operations.md` | `artifactOperations.ts` |
| `ConfigDict.md` | `types.ts#ConfigDict` |
| `Run.md` | `types.ts#Run` |
| `Artifact.md` | `types.ts#Artifact` |
| ... and more | ... |

## Benefits

- ✅ **Transparency** - Users can see the actual implementation
- ✅ **Deep Diving** - Developers can explore the source code
- ✅ **Reference** - Direct links for bug reports or contributions
- ✅ **Automatic** - Links are added during post-processing
- ✅ **Contextual** - Links include anchors for specific types

## How It Works

1. During post-processing, the script:
   - Maps each doc file to its likely source file
   - Adds a Source section before See Also
   - Creates GitHub URLs with the master branch

2. The mapping is based on TypeDoc's naming conventions:
   - Operations files → corresponding `.ts` files
   - Interface files → `types.ts` with anchors

## Future Improvements

Could enhance by:
- Extracting actual source file paths from TypeDoc metadata
- Supporting different GitHub branches
- Adding line number references
- Linking to specific function implementations

The source links make the documentation more valuable for developers who need to understand the implementation details!
