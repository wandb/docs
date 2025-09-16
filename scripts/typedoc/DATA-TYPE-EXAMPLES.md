# ✅ Comprehensive Structure Examples Added

## Feature
Every data type now has a clear TypeScript code example at the top showing the complete structure.

## What Was Added

### Complete Examples for All Types
Now all 10 data types have comprehensive examples:
- ✅ **Artifact** - Shows full artifact structure with metadata
- ✅ **ArtifactType** - Demonstrates artifact type definition
- ✅ **ArtifactVersion** - Version with alias, digest, and size
- ✅ **ConfigDict** - Configuration with various parameter types
- ✅ **Entity** - Team/user entity with members
- ✅ **Project** - Project with tags and counts
- ✅ **Run** - Complete run with config and summary
- ✅ **SummaryDict** - Summary metrics structure
- ✅ **Table** - Table with columns and rows
- ✅ **User** - User information structure

## Example Format

Each data type now starts with:
```markdown
**`Example`**

```typescript
const entity: Entity = {
  id: "entity_abc123",
  name: "my-team",
  isTeam: true,
  members: [
    { id: "user_1", username: "alice", role: "admin" },
    { id: "user_2", username: "bob", role: "member" }
  ],
  createdAt: new Date("2023-01-01"),
  updatedAt: new Date("2024-01-15")
};
```
```

## Benefits

1. **Immediate Understanding** - Developers can see the structure at a glance
2. **Copy-Paste Ready** - Examples can be used as templates
3. **Type Safety** - Shows proper TypeScript typing
4. **Real-World Values** - Uses realistic example data
5. **Complete Structure** - Shows all common properties

## Post-Processor Update

The `postprocess-hugo.js` now:
- Checks if data types have examples
- Adds default examples if missing
- Ensures consistent formatting
- Places examples after the description, before properties

## Result

Before:
```markdown
# Interface: Entity

Represents a W&B entity (team or individual user).

## Properties
...
```

After:
```markdown
# Interface: Entity

Represents a W&B entity (team or individual user).

**`Since`**

1.0.0

**`Example`**

```typescript
const entity: Entity = {
  id: "entity_abc123",
  name: "my-team",
  isTeam: true,
  members: [
    { id: "user_1", username: "alice", role: "admin" },
    { id: "user_2", username: "bob", role: "member" }
  ],
  createdAt: new Date("2023-01-01"),
  updatedAt: new Date("2024-01-15")
};
```

## Properties
...
```

## User Experience

Developers now get:
- **Quick overview** of what the type looks like
- **Working examples** they can adapt
- **Clear property relationships** 
- **Realistic data** to understand usage

This makes the documentation much more practical and immediately useful!
