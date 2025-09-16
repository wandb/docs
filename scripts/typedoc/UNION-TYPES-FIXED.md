# ✅ TypeScript Union Types Fixed

## What Was Wrong

Pipe characters `|` were being escaped as `\|` in TypeScript union types, both in code blocks and property descriptions.

## Examples Fixed

### In Function Signatures
**Before:**
```typescript
runJobType(run): string \| undefined
runLoggedArtifactVersion(run, artifactVersionName): ArtifactVersion \| undefined
```

**After:**
```typescript
runJobType(run): string | undefined
runLoggedArtifactVersion(run, artifactVersionName): ArtifactVersion | undefined
```

### In Property Descriptions
**Before:**
```markdown
• **state**: ``"running"`` \| ``"finished"`` \| ``"failed"`` \| ``"crashed"``
• **type**: ``"team"`` \| ``"user"``
```

**After:**
```markdown
• **state**: ``"running"`` | ``"finished"`` | ``"failed"`` | ``"crashed"``
• **type**: ``"team"`` | ``"user"``
```

## Why It Matters

- **Clean TypeScript**: Union types are a core TypeScript feature
- **No escaping needed**: Pipes don't need escaping in code blocks or inline code
- **Better readability**: Clean syntax is easier to read

## Post-Processor Updates

The `postprocess-hugo.js` script now:
1. Removes escaped pipes when converting signatures to code blocks
2. Cleans up any escaped pipes inside TypeScript code blocks
3. Fixes escaped pipes in property descriptions with inline code

## Results

- Fixed **2 function signatures** in Run_Operations.md
- Fixed **4 property descriptions** across data type files
- All union types now display correctly

The documentation now shows proper TypeScript syntax throughout!
