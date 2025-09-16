# ✅ Property Format Improved

## What Changed

### Old Format (cluttered)
```markdown
• **id**: string

Type ID
```

### New Format (clean)
```markdown
### id

#### Type

`string`

#### Description  

Type ID
```

## Benefits

### 1. Clear Structure
- Type and description are clearly separated
- Easier to scan and find information
- Consistent with professional API documentation

### 2. Better Readability
- No more bullet points and bold text mixed together
- Clean hierarchy with H4 subheadings
- Type stands out in code formatting

### 3. Proper Optional Handling
```markdown
#### Type

`string`

*Optional*

#### Description
```

## Fixed Issues

### Linked Types
**Before:** `` `[`ConfigDict`](../data-types/ConfigDict.md)` ``  
**After:** `[`ConfigDict`](../data-types/ConfigDict.md)`

### Union Types
**Before:** `` ``"team" | "user"`` ``  
**After:** `` `"team" | "user"` ``

## Results

- **54 properties** reformatted across 10 data type files
- All types now properly formatted
- Linked types render correctly
- Union types display cleanly

## Examples

### Simple Property
```markdown
### name

#### Type

`string`

#### Description

Artifact name
```

### Optional Property with Union Type
```markdown
### state

#### Type

`"running" | "finished" | "failed" | "crashed"`

*Optional*

#### Description

Current run state
```

### Linked Type Property
```markdown
### config

#### Type

[`ConfigDict`](../data-types/ConfigDict.md)

#### Description

Run configuration/hyperparameters
```

The property documentation is now clean, professional, and easy to read!
