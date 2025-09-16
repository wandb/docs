---
title: Run
---

A training or evaluation run logged to W&B.

```typescript
const run: Run = {
  id: "run_abc123",
  name: "sunny-dawn-42",
  state: "finished",
  config: {
    learning_rate: 0.001,
    batch_size: 32,
    epochs: 10
  },
  summaryMetrics: {
    loss: 0.023,
    accuracy: 0.95,
    val_accuracy: 0.93
  },
  createdAt: new Date("2024-01-15T10:30:00Z"),
  updatedAt: new Date("2024-01-15T14:45:00Z")
};
```

| Property | Type | Description |
| :------- | :--- | :---------- |
| `id` | `string` | Run ID |
| `name` | `string` | Run name |
| `state` | `string` | Run state (e.g., "running", "finished", "failed") |
| `config` | [`ConfigDict`](../data-types/configdict.md) | *Optional*. Run configuration |
| `summaryMetrics` | [`SummaryDict`](../data-types/summarydict.md) | *Optional*. Summary metrics |
| `createdAt` | `Date` | Creation timestamp |
| `updatedAt` | `Date` | Last update timestamp |
