# ✅ Fixed Union Types Breaking Tables

## The Problem

Markdown tables were breaking because pipe characters `|` in union types were being interpreted as column separators:

```markdown
| `type` | `"team" | "user"` | Entity type |
```

This was rendering as **4 columns** instead of 3 because the pipe in the union type was splitting the cell!

## The Solution

Escape the pipes within union types (but NOT the table column separators):

```markdown
| `type` | `"team" \| "user"` | Entity type |
```

## What Was Fixed

### 4 Union Types in Tables:
1. **Entity.md**: `"team" | "user"` → `"team" \| "user"`
2. **Project.md**: `"public" | "private"` → `"public" \| "private"`
3. **Run.md**: `"running" | "finished" | "failed" | "crashed"` → properly escaped
4. **SummaryDict.md**: Same state union type → properly escaped

## Why It Broke

When converting properties to table format, union types with pipes weren't escaped, so markdown interpreted them as column separators.

## Key Learning

In markdown tables, any literal pipe character `|` that's NOT a column separator must be escaped as `\|`.

### Example:
```markdown
✅ CORRECT:
| Property | Type | Description |
| :------- | :--- | :---------- |
| `state` | `"running" \| "finished"` | Run state |

❌ WRONG (creates 4 columns):
| Property | Type | Description |
| :------- | :--- | :---------- |
| `state` | `"running" | "finished"` | Run state |
```

## Result

Tables now render correctly with exactly 3 columns:
- Property name
- Type (with properly escaped union types)
- Description

The documentation tables are now clean and properly formatted!
