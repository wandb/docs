#!/usr/bin/env node

/**
 * Update ConfigDict and SummaryDict with practical, user-focused examples
 */

const fs = require('fs');
const path = require('path');

const dataTypesDir = path.join(__dirname, '../content/en/ref/query-panel-new/data-types');

console.log('Updating ConfigDict and SummaryDict with practical examples...\n');

// ConfigDict with practical examples
const configDictContent = `---
title: ConfigDict
---

Configuration dictionary for W&B runs. Stores hyperparameters, settings, and metadata.

\`\`\`typescript
// Typical ML configuration:
const config: ConfigDict = {
  // Training hyperparameters
  learning_rate: 0.001,
  batch_size: 32,
  epochs: 100,
  optimizer: "adam",
  
  // Model architecture
  model_name: "resnet50",
  num_layers: 50,
  dropout_rate: 0.2,
  hidden_dims: [512, 256, 128],
  
  // Data settings
  dataset: "imagenet",
  augmentation: true,
  train_split: 0.8,
  
  // System settings
  device: "cuda",
  num_workers: 4,
  seed: 42
};
\`\`\`

## Value Types

**Basic Types**
- \`string\`: Text values like model names, optimizers, datasets
- \`number\`: Numeric values including integers, floats, and scientific notation
- \`boolean\`: True/false flags
- \`null\`: Null values for optional settings

**Complex Types**
- \`Array\`: Lists of any allowed type (e.g., \`[512, 256, 128]\`)
- \`Object\`: Nested configuration groups with string keys

**Special W&B Types** (automatically handled when logging)
- W&B Tables: When you log \`wandb.Table()\`, appears as table reference
- W&B Artifacts: When referencing artifacts, appears as artifact link

## Common Patterns

\`\`\`typescript
// Nested configuration groups
const config = {
  optimizer: {
    type: "adam",
    betas: [0.9, 0.999],
    weight_decay: 0.0001
  },
  scheduler: {
    type: "cosine",
    warmup_steps: 1000
  }
};

// Environment and metadata
const config = {
  experiment_name: "baseline_v2",
  git_commit: "abc123def",
  python_version: "3.9.7",
  cuda_version: "11.8"
};
\`\`\`

## Constraints
- Keys must be strings
- Values must be JSON-serializable
- Keys starting with \`_wandb\` are reserved
- No functions, undefined, or symbols allowed
`;

// SummaryDict with practical examples
const summaryDictContent = `---
title: SummaryDict
---

Summary dictionary for W&B runs. Stores final metrics, best values, and aggregated results.

\`\`\`typescript
// Typical training summary:
const summary: SummaryDict = {
  // Final metrics
  final_loss: 0.0234,
  final_accuracy: 0.9523,
  
  // Best values during training
  best_val_loss: 0.0198,
  best_val_accuracy: 0.9612,
  best_epoch: 87,
  
  // Training statistics
  total_train_time: 3600.5,  // seconds
  total_steps: 50000,
  early_stopped: false,
  
  // Test set results
  test_accuracy: 0.9487,
  test_f1_score: 0.9465
};
\`\`\`

## Value Types

**Basic Types**
- \`string\`: Text summaries, model paths, status messages
- \`number\`: Metrics, scores, counts, durations
- \`boolean\`: Binary flags like convergence status
- \`null\`: For optional metrics that weren't computed

**Complex Types**
- \`Array\`: Lists like per-class scores (e.g., \`[0.92, 0.94, 0.96]\`)
- \`Object\`: Grouped metrics with string keys

**Special W&B Types** (automatically handled when logging)
- W&B Histograms: Distribution visualizations
- W&B Tables: Detailed results tables

## Common Patterns

\`\`\`typescript
// Grouped metrics by dataset split
const summary = {
  train: {
    loss: 0.023,
    accuracy: 0.975,
    samples_seen: 50000
  },
  validation: {
    loss: 0.045,
    accuracy: 0.948,
    samples_seen: 10000
  },
  test: {
    loss: 0.041,
    accuracy: 0.951,
    samples_seen: 10000
  }
};

// Multi-class classification results
const summary = {
  accuracy: 0.92,
  macro_f1: 0.91,
  per_class_precision: [0.95, 0.89, 0.92, 0.90],
  per_class_recall: [0.93, 0.91, 0.90, 0.92],
  confusion_matrix_logged: true  // Actual matrix logged as W&B Table
};

// Model information
const summary = {
  total_parameters: 125_000_000,
  trainable_parameters: 124_500_000,  
  model_size_mb: 476.8,
  inference_time_ms: 23.4
};
\`\`\`

## Constraints
- Keys must be strings
- Values must be JSON-serializable
- Keys starting with \`_wandb\` are reserved
- Special: Supports NaN for missing/invalid metrics
- No functions, undefined, or symbols allowed
`;

// Write the updated files
fs.writeFileSync(path.join(dataTypesDir, 'ConfigDict.md'), configDictContent);
console.log('✅ Updated ConfigDict.md with practical examples');

fs.writeFileSync(path.join(dataTypesDir, 'SummaryDict.md'), summaryDictContent);  
console.log('✅ Updated SummaryDict.md with practical examples');

console.log('\n✨ Done! Both types now have practical, user-focused documentation.');
