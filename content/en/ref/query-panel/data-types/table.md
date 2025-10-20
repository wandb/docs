---
title: Table
---

W&B Table for structured data logging and visualization.

```typescript
const table: Table = {
  columns: ["epoch", "loss", "accuracy"],
  data: [
    [1, 0.5, 0.75],
    [2, 0.3, 0.85],
    [3, 0.2, 0.90]
  ]
};
```

| Property | Type | Description |
| :------- | :--- | :---------- |
| `columns` | `string[]` | Column names |
| `data` | `any[][]` | Table data rows |
