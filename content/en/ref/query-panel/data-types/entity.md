---
title: Entity
---

Represents a W&B entity (team or individual user).

```typescript
const entity: Entity = {
  id: "entity_abc123",
  name: "my-team",
  isTeam: true
};
```

| Property | Type | Description |
| :------- | :--- | :---------- |
| `id` | `string` | Entity ID |
| `name` | `string` | Entity name |
| `isTeam` | `boolean` | Whether this is a team or individual user |
