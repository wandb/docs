---
title: ConfigDict
---

Configuration dictionary for W&B runs. Stores hyperparameters, settings, and metadata.

```typescript
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
```

## Value Types

**Basic Types**
- `string`: Text values like model names, optimizers, datasets
- `number`: Numeric values including integers, floats, and scientific notation
- `boolean`: True/false flags
- `null`: Null values for optional settings

**Complex Types**
- `Array`: Lists of any allowed type (e.g., `[512, 256, 128]`)
- `Object`: Nested configuration groups with string keys

**Special W&B Types** (automatically handled when logging)
- W&B Tables: Appear as reference objects with table metadata (columns, rows, path)
- W&B Artifacts: Appear as reference objects with version and ID information

```typescript
// What you see after logging a W&B Table to config:
const config: ConfigDict = {
  // ... your normal config ...
  
  // This appears automatically when you log wandb.Table() to config:
  "evaluation_results": {
    "_type": "table-file",
    "ncols": 5,
    "nrows": 100,
    "path": "media/table/eval_results_2_abc123.table.json"
  }
};

// What you see after referencing an artifact in config:
const config: ConfigDict = {
  // ... your normal config ...
  
  // This appears when you use an artifact:
  "model_artifact": {
    "_type": "artifactVersion",
    "id": "QXJ0aWZhY3Q6MTIzNDU2",
    "version": "v3",
    "path": "model-weights:v3"
  }
};
```

## Common Patterns

```typescript
// Nested configuration groups
const config: ConfigDict = {
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
const config: ConfigDict = {
  experiment_name: "baseline_v2",
  git_commit: "abc123def",
  python_version: "3.9.7",
  cuda_version: "11.8"
};
```

## Constraints
- Keys must be strings
- Values must be JSON-serializable
- Keys starting with `_wandb` are reserved
- No functions, undefined, or symbols allowed
