#!/usr/bin/env node

/**
 * Fix all data types to match the actual TypeScript source
 * Based on inspection of core/frontends/weave/src/core/ops/domain/*.ts
 */

const fs = require('fs');
const path = require('path');

const dataTypesDir = path.join(__dirname, '../content/en/ref/query-panel-new/data-types');

console.log('Fixing all data types to match actual source code...\n');

// Entity - uses isTeam boolean, not type string
const entityContent = `---
title: Entity
---

Represents a W&B entity (team or individual user).

\`\`\`typescript
const entity: Entity = {
  id: "entity_abc123",
  name: "my-team",
  isTeam: true
};
\`\`\`

| Property | Type | Description |
| :------- | :--- | :---------- |
| \`id\` | \`string\` | Entity ID |
| \`name\` | \`string\` | Entity name |
| \`isTeam\` | \`boolean\` | Whether this is a team or individual user |
`;

// User
const userContent = `---
title: User
---

Represents a W&B user.

\`\`\`typescript
const user: User = {
  id: "user_123",
  username: "john_doe",
  name: "John Doe",
  email: "john@example.com"
};
\`\`\`

| Property | Type | Description |
| :------- | :--- | :---------- |
| \`id\` | \`string\` | User ID |
| \`username\` | \`string\` | Username |
| \`name\` | \`string\` | *Optional*. User's full name |
| \`email\` | \`string\` | *Optional*. User email |
`;

// ArtifactType
const artifactTypeContent = `---
title: ArtifactType
---

Artifact type definition.

\`\`\`typescript
const artifactType: ArtifactType = {
  name: "model"
};
\`\`\`

| Property | Type | Description |
| :------- | :--- | :---------- |
| \`name\` | \`string\` | Type name |
`;

// Artifact
const artifactContent = `---
title: Artifact
---

W&B artifact for versioning datasets, models, and other files.

\`\`\`typescript
const artifact: Artifact = {
  id: "artifact_abc123",
  name: "model-weights",
  type: artifactType,
  description: "Trained model weights",
  aliases: ["latest", "production"],
  createdAt: new Date("2024-01-15")
};
\`\`\`

| Property | Type | Description |
| :------- | :--- | :---------- |
| \`id\` | \`string\` | Artifact ID |
| \`name\` | \`string\` | Artifact name |
| \`type\` | \`ArtifactType\` | Artifact type |
| \`description\` | \`string\` | *Optional*. Artifact description |
| \`aliases\` | \`string[]\` | *Optional*. List of aliases |
| \`createdAt\` | \`Date\` | Creation timestamp |
`;

// ArtifactVersion
const artifactVersionContent = `---
title: ArtifactVersion
---

A specific version of a W&B artifact.

\`\`\`typescript
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
\`\`\`

| Property | Type | Description |
| :------- | :--- | :---------- |
| \`id\` | \`string\` | Version ID |
| \`version\` | \`string\` | Version string (e.g., "v3") |
| \`versionIndex\` | \`number\` | Version index number |
| \`aliases\` | \`string[]\` | *Optional*. List of aliases |
| \`createdAt\` | \`Date\` | Creation timestamp |
| \`metadata\` | \`object\` | *Optional*. Version metadata |
`;

// Project
const projectContent = `---
title: Project
---

W&B project containing runs, artifacts, and reports.

\`\`\`typescript
const project: Project = {
  name: "my-awesome-project",
  entity: entity,
  createdAt: new Date("2023-01-01"),
  updatedAt: new Date("2024-01-20")
};
\`\`\`

| Property | Type | Description |
| :------- | :--- | :---------- |
| \`name\` | \`string\` | Project name |
| \`entity\` | \`Entity\` | Owning entity |
| \`createdAt\` | \`Date\` | Creation timestamp |
| \`updatedAt\` | \`Date\` | Last update timestamp |
`;

// Run
const runContent = `---
title: Run
---

A training or evaluation run logged to W&B.

\`\`\`typescript
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
\`\`\`

| Property | Type | Description |
| :------- | :--- | :---------- |
| \`id\` | \`string\` | Run ID |
| \`name\` | \`string\` | Run name |
| \`state\` | \`string\` | Run state (e.g., "running", "finished", "failed") |
| \`config\` | \`ConfigDict\` | *Optional*. Run configuration |
| \`summaryMetrics\` | \`SummaryDict\` | *Optional*. Summary metrics |
| \`createdAt\` | \`Date\` | Creation timestamp |
| \`updatedAt\` | \`Date\` | Last update timestamp |
`;

// ConfigDict
const configDictContent = `---
title: ConfigDict
---

Configuration dictionary for W&B runs.

\`\`\`typescript
const config: ConfigDict = {
  learning_rate: 0.001,
  batch_size: 32,
  epochs: 10,
  optimizer: "adam",
  model: {
    layers: 12,
    hidden_dim: 768,
    dropout: 0.1
  }
};
\`\`\`

| Property | Type | Description |
| :------- | :--- | :---------- |
| \`[key: string]\` | \`any\` | Configuration values can be any JSON-serializable type |
`;

// SummaryDict  
const summaryDictContent = `---
title: SummaryDict
---

Summary metrics dictionary for W&B runs.

\`\`\`typescript
const summary: SummaryDict = {
  loss: 0.023,
  accuracy: 0.95,
  val_accuracy: 0.93,
  total_steps: 10000,
  best_epoch: 8
};
\`\`\`

| Property | Type | Description |
| :------- | :--- | :---------- |
| \`[key: string]\` | \`any\` | Summary values can be numbers, strings, or nested objects |
`;

// Table
const tableContent = `---
title: Table
---

W&B Table for structured data logging and visualization.

\`\`\`typescript
const table: Table = {
  columns: ["epoch", "loss", "accuracy"],
  data: [
    [1, 0.5, 0.75],
    [2, 0.3, 0.85],
    [3, 0.2, 0.90]
  ]
};
\`\`\`

| Property | Type | Description |
| :------- | :--- | :---------- |
| \`columns\` | \`string[]\` | Column names |
| \`data\` | \`any[][]\` | Table data rows |
`;

// Write all files
const files = {
  'Entity.md': entityContent,
  'User.md': userContent,
  'ArtifactType.md': artifactTypeContent,
  'Artifact.md': artifactContent,
  'ArtifactVersion.md': artifactVersionContent,
  'Project.md': projectContent,
  'Run.md': runContent,
  'ConfigDict.md': configDictContent,
  'SummaryDict.md': summaryDictContent,
  'Table.md': tableContent
};

Object.entries(files).forEach(([filename, content]) => {
  const filePath = path.join(dataTypesDir, filename);
  fs.writeFileSync(filePath, content);
  console.log(`✅ Fixed ${filename}`);
});

console.log('\n✨ All data types have been corrected to match the actual source code!');
console.log('\n📝 Note: These are now accurate based on the TypeScript operations in:');
console.log('   core/frontends/weave/src/core/ops/domain/*.ts');
