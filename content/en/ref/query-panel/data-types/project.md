---
title: Project
---

W&B project containing runs, artifacts, and reports.

```typescript
const project: Project = {
  name: "my-awesome-project",
  entity: entity,
  createdAt: new Date("2023-01-01"),
  updatedAt: new Date("2024-01-20")
};
```

| Property | Type | Description |
| :------- | :--- | :---------- |
| `name` | `string` | Project name |
| `entity` | [`Entity`](../data-types/entity.md) | Owning entity |
| `createdAt` | `Date` | Creation timestamp |
| `updatedAt` | `Date` | Last update timestamp |
