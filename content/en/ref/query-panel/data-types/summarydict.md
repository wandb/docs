---
title: SummaryDict
---

Summary dictionary for W&B runs. Stores final metrics, best values, and aggregated results.

```typescript
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
```

## Value Types

**Basic Types**
- `string`: Text summaries, model paths, status messages
- `number`: Metrics, scores, counts, durations
- `boolean`: Binary flags like convergence status
- `null`: For optional metrics that weren't computed

**Complex Types**
- `Array`: Lists like per-class scores (e.g., `[0.92, 0.94, 0.96]`)
- `Object`: Grouped metrics with string keys

**Special W&B Types** (automatically handled when logging)
- W&B Histograms: Appear as objects with bins and values arrays
- W&B Tables: Appear as reference objects with table metadata (columns, rows, path)
- W&B Artifacts: Appear as reference objects with version and ID information

```typescript
// What you see after logging W&B special types to summary:
const summary: SummaryDict = {
  // ... your normal metrics ...
  
  // This appears when you log wandb.Histogram():
  "weight_distribution": {
    "_type": "histogram",
    "bins": [0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "values": [10, 25, 45, 30, 15, 5]
  },
  
  // This appears when you log wandb.Table():
  "predictions_table": {
    "_type": "table-file", 
    "ncols": 4,
    "nrows": 1000,
    "path": "media/table/predictions_3_def456.table.json"
  },
  
  // This appears when you reference an artifact:
  "best_model": {
    "_type": "artifactVersion",
    "id": "QXJ0aWZhY3Q6OTg3NjU0",
    "version": "v12",
    "path": "model-checkpoint:v12"
  }
};
```

## Common Patterns

```typescript
// Grouped metrics by dataset split
const summary: SummaryDict = {
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
const summary: SummaryDict = {
  accuracy: 0.92,
  macro_f1: 0.91,
  per_class_precision: [0.95, 0.89, 0.92, 0.90],
  per_class_recall: [0.93, 0.91, 0.90, 0.92],
  confusion_matrix_logged: true  // Actual matrix logged as W&B Table
};

// Model information
const summary: SummaryDict = {
  total_parameters: 125_000_000,
  trainable_parameters: 124_500_000,  
  model_size_mb: 476.8,
  inference_time_ms: 23.4
};
```

## Constraints
- Keys must be strings
- Values must be JSON-serializable
- Keys starting with `_wandb` are reserved
- Special: Supports NaN for missing/invalid metrics
- No functions, undefined, or symbols allowed
