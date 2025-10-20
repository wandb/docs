---
title: Artifact
---

W&B artifact for versioning datasets, models, and other files.

```typescript
const artifact: Artifact = {
  id: "artifact_abc123",
  name: "model-weights",
  type: artifactType,
  description: "Trained model weights",
  aliases: ["latest", "production"],
  createdAt: new Date("2024-01-15")
};
```

| Property | Type | Description |
| :------- | :--- | :---------- |
| `id` | `string` | Artifact ID |
| `name` | `string` | Artifact name |
| `type` | [`ArtifactType`](../data-types/artifacttype.md) | Artifact type |
| `description` | `string` | *Optional*. Artifact description |
| `aliases` | `string[]` | *Optional*. List of aliases |
| `createdAt` | `Date` | Creation timestamp |
