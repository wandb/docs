# ✅ Filename Cleanup Complete

## What Changed

Removed the redundant `W_B_Query_Expression_Language.` prefix from all data type files.

### Before
```
data-types/
├── W_B_Query_Expression_Language.Run.md
├── W_B_Query_Expression_Language.Artifact.md
├── W_B_Query_Expression_Language.ConfigDict.md
└── ...
```

### After
```
data-types/
├── Run.md
├── Artifact.md
├── ConfigDict.md
├── SummaryDict.md
├── Table.md
├── Entity.md
├── Project.md
├── User.md
├── ArtifactType.md
└── ArtifactVersion.md
```

## Script Updates

The `postprocess-hugo.js` script now:
1. **Automatically removes the prefix** when moving interface files to data-types/
2. **Cleans up titles** to remove the prefix if present
3. **Generates clean _index.md** with proper links to renamed files

## Benefits

- ✅ **Cleaner filenames** - No redundant prefixes
- ✅ **Better readability** - Immediately clear what each file contains
- ✅ **Cleaner URLs** - `/data-types/Run` instead of `/data-types/W_B_Query_Expression_Language.Run`
- ✅ **Automatic handling** - Script handles renaming in future generations

The context is already clear from the directory structure (`query-panel-new/data-types/`), so the prefix was unnecessary noise!
