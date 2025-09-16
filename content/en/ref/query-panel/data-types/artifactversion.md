---
title: ArtifactVersion
---

A specific version of a W&B artifact.

```typescript
const artifactVersion: ArtifactVersion = {
  id: "version_xyz789",
  version: "v3",
  versionIndex: 3,
  aliases: ["latest", "production"],
  createdAt: new Date("2024-01-15"),
  metadata: {
    accuracy: 0.95,
    model_type: "transformer"
  }
};
```

| Property | Type | Description |
| :------- | :--- | :---------- |
| `id` | `string` | Version ID |
| `version` | `string` | Version string (e.g., "v3") |
| `versionIndex` | `number` | Version index number |
| `aliases` | `string[]` | *Optional*. List of aliases |
| `createdAt` | `Date` | Creation timestamp |
| `metadata` | `object` | *Optional*. Version metadata |
