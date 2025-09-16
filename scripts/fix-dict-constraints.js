#!/usr/bin/env node

/**
 * Fix ConfigDict and SummaryDict to document actual constraints
 */

const fs = require('fs');
const path = require('path');

const dataTypesDir = path.join(__dirname, '../content/en/ref/query-panel-new/data-types');

console.log('Updating ConfigDict and SummaryDict with actual constraints...\n');

// ConfigDict with constraints
const configDictContent = `---
title: ConfigDict
---

Configuration dictionary for W&B runs. Accepts any JSON-serializable values with some constraints.

\`\`\`typescript
// Example with common configuration fields:
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

// Nested structures and arrays are supported:
const advancedConfig: ConfigDict = {
  experiment_name: "test-123",
  use_cuda: true,
  seed: 42,
  layer_sizes: [768, 512, 256, 128],
  hyperparameters: {
    learning_rate: {
      initial: 0.001,
      schedule: "cosine",
      min_lr: 0.00001
    }
  }
};
\`\`\`

## Allowed Value Types

| Type | Example | Notes |
| :--- | :------ | :---- |
| \`string\` | \`"adam"\`, \`"bert-base"\` | Any string value |
| \`number\` | \`0.001\`, \`42\`, \`3.14\` | Integers, floats, NaN |
| \`boolean\` | \`true\`, \`false\` | Boolean values |
| \`null\` | \`null\` | Null values |
| \`Array\` | \`[1, 2, 3]\`, \`["a", "b"]\` | Arrays of any allowed types |
| \`Object\` | \`{ nested: "value" }\` | Nested objects with string keys |
| W&B Table | Special W&B type | Logged tables appear as special references |
| W&B Artifact | Special W&B type | Artifact references in config |

## Constraints

- ❌ **Not allowed**: Functions, undefined, symbols, or other non-JSON types
- ❌ **Reserved keys**: Keys starting with \`_wandb\` are reserved for internal use
- ✅ **Keys**: Must be strings
- ✅ **Nesting**: Arbitrary nesting depth is supported
`;

// SummaryDict with constraints
const summaryDictContent = `---
title: SummaryDict
---

Summary metrics dictionary for W&B runs. Accepts any JSON-serializable values with some constraints.

\`\`\`typescript
// Example with common summary metrics:
const summary: SummaryDict = {
  loss: 0.023,
  accuracy: 0.95,
  val_accuracy: 0.93,
  total_steps: 10000,
  best_epoch: 8
};

// Complex nested structures are supported:
const detailedSummary: SummaryDict = {
  train: {
    loss: 0.023,
    accuracy: 0.95,
    per_class_accuracy: [0.92, 0.94, 0.96, 0.95]
  },
  validation: {
    loss: 0.045,
    accuracy: 0.93
  },
  model_info: {
    total_params: 125000000,
    trainable_params: 124500000
  },
  best_checkpoint: "checkpoint-8000",
  completed: true
};
\`\`\`

## Allowed Value Types

| Type | Example | Notes |
| :--- | :------ | :---- |
| \`string\` | \`"completed"\`, \`"checkpoint-8000"\` | Any string value |
| \`number\` | \`0.95\`, \`10000\`, \`2.5e-4\` | Integers, floats, NaN |
| \`boolean\` | \`true\`, \`false\` | Boolean values |
| \`null\` | \`null\` | Null values |
| \`Array\` | \`[0.92, 0.94, 0.96]\` | Arrays of any allowed types |
| \`Object\` | \`{ train: {...} }\` | Nested objects with string keys |
| W&B Table | Special W&B type | Tables logged as summary values |
| W&B Histogram | Special W&B type | Histograms in summary |

## Constraints

- ❌ **Not allowed**: Functions, undefined, symbols, or other non-JSON types
- ❌ **Reserved keys**: Keys starting with \`_wandb\` are reserved for internal use
- ✅ **Keys**: Must be strings
- ✅ **Nesting**: Arbitrary nesting depth is supported
- ✅ **Special values**: NaN is supported (unlike standard JSON)
`;

// Write the updated files
fs.writeFileSync(path.join(dataTypesDir, 'ConfigDict.md'), configDictContent);
console.log('✅ Updated ConfigDict.md with actual constraints');

fs.writeFileSync(path.join(dataTypesDir, 'SummaryDict.md'), summaryDictContent);  
console.log('✅ Updated SummaryDict.md with actual constraints');

console.log('\n✨ Done! Both dictionary types now document the actual allowed types and constraints.');
