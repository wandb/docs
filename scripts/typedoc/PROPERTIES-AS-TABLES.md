# ✅ Properties Now in Clean Table Format

## The Problem

The previous format with separate Type and Description headers was creating unnecessarily long pages:

```markdown
### id

#### Type

`string`

#### Description

Type ID

___

### name

#### Type

`string`

#### Description

Type name

___

(... and so on for many properties ...)
```

## The Solution: Clean Tables

Now all properties are in a compact, scannable table:

```markdown
## Properties

| Property | Type | Description |
| :------- | :--- | :---------- |
| `id` | `string` | Type ID |
| `name` | `string` | Type name |
| `description` | `string` | *Optional*. Type description |
```

## Benefits

### 📊 **50-70% Shorter Pages**
- ArtifactType.md: From 56 lines → ~26 lines
- Run.md: From 145 lines → ~36 lines for properties

### 👁️ **Much Easier to Scan**
- All information visible at a glance
- No scrolling through repetitive headers
- Clean alignment makes comparison easy

### 🔗 **Handles All Type Formats**
- Simple types: `` `string` ``
- Linked types: `[`ConfigDict`](../data-types/ConfigDict.md)`
- Union types: `` `"running" | "finished" | "failed"` ``
- Optional markers: `*Optional*. Description here`

## Examples

### Simple Data Type (User)
```markdown
| Property | Type | Description |
| :------- | :--- | :---------- |
| `id` | `string` | User ID |
| `username` | `string` | Username |
| `email` | `string` | *Optional*. User email |
```

### Complex Data Type (Run)
```markdown
| Property | Type | Description |
| :------- | :--- | :---------- |
| `id` | `string` | Unique run identifier |
| `name` | `string` | Run display name |
| `config` | [`ConfigDict`](../data-types/ConfigDict.md) | Run configuration/hyperparameters |
| `state` | `"running" | "finished" | "failed" | "crashed"` | *Optional*. Current run state |
| `user` | [`User`](../data-types/User.md) | *Optional*. User who created the run |
```

## Results

✅ **54 properties** across 10 data types converted to table format
✅ Pages are now **much shorter** and easier to read
✅ Professional API documentation style
✅ Consistent with modern documentation standards

## Comparison

### Before (Long)
- Multiple H3 and H4 headers per property
- Lots of vertical space
- Hard to compare properties
- ~150+ lines for complex types

### After (Compact)
- Single clean table
- All info visible at once
- Easy to scan and compare
- ~30-40 lines for complex types

The documentation is now much more user-friendly and professional!
