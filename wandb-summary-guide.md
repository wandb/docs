# W&B Summary Metrics: Comprehensive Guide

## Overview

W&B Summary metrics provide a powerful way to capture and display key performance indicators from your machine learning experiments. Unlike logged metrics that track values over time, summary metrics represent single, definitive values that characterize your entire run.

## What are Summary Metrics?

Summary metrics in W&B are:
- **Single values** that summarize a model's performance or a preprocessing step
- **Automatically updated** with the last logged value by default
- **Customizable** to capture minimum, maximum, mean, or best values
- **Displayed prominently** in the runs table and run overview pages
- **Queryable** via the W&B API for programmatic analysis

## Key Differences: `log()` vs `summary`

| Feature | `wandb.log()` | `wandb.summary` |
|---------|---------------|-----------------|
| **Purpose** | Track metrics over time | Store final/best values |
| **Display** | Line plots, time series | Tables, overview panels |
| **Values** | Multiple values per metric | Single value per metric |
| **Update frequency** | Every step/epoch | Once or when explicitly set |
| **Use case** | Training curves | Model comparison |

## How to Use Summary Metrics

### 1. Automatic Summary Updates

By default, the last value logged with `wandb.log()` becomes the summary value:

```python
import wandb

with wandb.init() as run:
    for epoch in range(10):
        accuracy = train_epoch()
        run.log({"accuracy": accuracy})
    # The summary will contain the final accuracy value
```

### 2. Manual Summary Updates

Set custom summary values to capture important metrics:

```python
import wandb

with wandb.init() as run:
    best_accuracy = 0
    for epoch in range(1, num_epochs + 1):
        test_loss, test_accuracy = test()
        run.log({"epoch": epoch, "test_accuracy": test_accuracy})
        
        if test_accuracy > best_accuracy:
            run.summary["best_accuracy"] = test_accuracy
            best_accuracy = test_accuracy
```

### 3. Using `define_metric` for Automatic Summary Calculation

Configure automatic summary calculations for specific metrics:

```python
import wandb
import random

with wandb.init() as run:
    # Configure automatic min/max tracking
    run.define_metric("loss", summary="min")
    run.define_metric("accuracy", summary="max")
    
    for i in range(100):
        run.log({
            "loss": random.uniform(0, 1 / (i + 1)),
            "accuracy": random.uniform(1 / (i + 1), 1)
        })
    # Summary will contain min loss and max accuracy
```

Available summary options:
- `"min"` - Minimum value logged
- `"max"` - Maximum value logged
- `"mean"` - Average of all logged values
- `"best"` - Best value (requires `objective` parameter)
- `"last"` - Last logged value (default)
- `"none"` - Don't create a summary metric

### 4. Complex Data Types in Summary

Summary can handle various data types:

```python
import wandb
import numpy as np

with wandb.init() as run:
    # Arrays and tensors
    run.summary["final_weights"] = np.random.random((100, 100))
    
    # Custom metrics
    run.summary["model_size_mb"] = model_size / 1024 / 1024
    run.summary["training_time_hours"] = total_time / 3600
    
    # Composite metrics
    run.summary["efficiency_score"] = accuracy / training_time
```

### 5. Updating Summary Post-Training

Use the W&B API to update summaries after a run completes:

```python
import wandb

# Access a completed run
api = wandb.Api()
run = api.run("entity/project/run_id")

# Update summary values
run.summary["post_process_accuracy"] = 0.95
run.summary["final_evaluation_score"] = 0.88
run.summary.update()
```

## Best Practices

### 1. Track Key Performance Indicators

```python
with wandb.init() as run:
    # Training metrics
    run.summary["final_train_loss"] = final_train_loss
    run.summary["best_val_accuracy"] = best_val_accuracy
    
    # Resource utilization
    run.summary["total_gpu_hours"] = gpu_hours
    run.summary["peak_memory_gb"] = peak_memory
    
    # Model characteristics
    run.summary["total_parameters"] = count_parameters(model)
    run.summary["model_flops"] = calculate_flops(model)
```

### 2. Multi-Metric Optimization

When optimizing multiple metrics, create weighted combinations:

```python
with wandb.init() as run:
    # Log individual metrics
    accuracy = run.summary.get("accuracy", 0.0)
    f1_score = run.summary.get("f1_score", 0.0)
    inference_time = run.summary.get("inference_time", 1.0)
    
    # Create composite metric
    # Higher is better: accuracy and f1_score
    # Lower is better: inference_time (so we use reciprocal)
    composite_score = (0.4 * accuracy + 0.4 * f1_score + 0.2 * (1/inference_time))
    run.summary["composite_score"] = composite_score
```

### 3. Hyperparameter Search Summary

For hyperparameter sweeps, use summary metrics to compare runs:

```python
import wandb

sweep_config = {
    'metric': {
        'name': 'best_val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {'values': [0.001, 0.01, 0.1]},
        'batch_size': {'values': [16, 32, 64]}
    }
}

def train():
    with wandb.init() as run:
        config = wandb.config
        best_val_acc = 0
        
        for epoch in range(num_epochs):
            val_acc = validate()
            run.log({"val_accuracy": val_acc, "epoch": epoch})
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                run.summary["best_val_accuracy"] = best_val_acc
                run.summary["best_epoch"] = epoch

sweep_id = wandb.sweep(sweep_config, project="my-project")
wandb.agent(sweep_id, train, count=10)
```

## Common Use Cases

### 1. Model Comparison Dashboard

Summary metrics enable easy model comparison in the runs table:

```python
# Log consistent summary metrics across all experiments
run.summary.update({
    "final_accuracy": final_acc,
    "best_accuracy": best_acc,
    "training_time": time.time() - start_time,
    "model_size_mb": get_model_size(model),
    "inference_latency_ms": measure_latency(model)
})
```

### 2. Early Stopping Integration

```python
with wandb.init() as run:
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(max_epochs):
        val_loss = validate()
        run.log({"val_loss": val_loss})
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            run.summary["best_val_loss"] = best_val_loss
            run.summary["best_epoch"] = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter > patience:
            run.summary["early_stopped"] = True
            run.summary["stopped_epoch"] = epoch
            break
```

### 3. A/B Testing Results

```python
with wandb.init() as run:
    # Run A/B test
    results_a, results_b = run_ab_test()
    
    # Log summary statistics
    run.summary.update({
        "variant_a_conversion": results_a.conversion_rate,
        "variant_b_conversion": results_b.conversion_rate,
        "improvement_percentage": (results_b.conversion_rate - results_a.conversion_rate) / results_a.conversion_rate * 100,
        "statistical_significance": results_a.p_value < 0.05,
        "sample_size_a": len(results_a.samples),
        "sample_size_b": len(results_b.samples)
    })
```

## Accessing Summary Metrics

### 1. In the W&B UI

- **Run Overview Tab**: View all summary metrics in the Summary section
- **Runs Table**: Sort and filter runs by summary values
- **Charts**: Use summary values in scatter plots and parallel coordinates

### 2. Via the API

```python
import wandb
import pandas as pd

api = wandb.Api()

# Get runs from a project
runs = api.runs("entity/project")

# Extract summary metrics into a DataFrame
summary_list = []
for run in runs:
    summary_list.append({
        "name": run.name,
        "id": run.id,
        **run.summary._json_dict
    })

runs_df = pd.DataFrame(summary_list)

# Analysis examples
best_run = runs_df.loc[runs_df['best_accuracy'].idxmax()]
print(f"Best performing run: {best_run['name']} with accuracy: {best_run['best_accuracy']}")
```

### 3. In Reports

Reference summary metrics in W&B Reports:

```python
# In report panels, reference summary metrics
# Panel expression examples:
# ${run:summary.best_accuracy}
# ${run:summary.training_time}
```

## Technical Details

### Storage and Performance

- Summary metrics are stored separately from logged history
- They don't impact storage quotas significantly
- Tensors and arrays are stored as binary files with statistical summaries
- Summary updates are atomic operations

### Limitations and Considerations

1. **One value per metric**: Summary can only store one value per metric name
2. **Overwriting**: Setting a summary value overwrites the previous value
3. **No history**: Summary doesn't maintain a history of changes
4. **Size limits**: Large tensors in summary may impact UI loading times

## Integration Examples

### PyTorch Lightning

```python
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.best_val_accuracy = 0
        
    def validation_epoch_end(self, outputs):
        val_accuracy = self.compute_accuracy(outputs)
        self.log("val_accuracy", val_accuracy)
        
        if val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = val_accuracy
            self.logger.experiment.summary["best_val_accuracy"] = val_accuracy
```

### TensorFlow/Keras

```python
import wandb
from wandb.keras import WandbCallback

class SummaryCallback(WandbCallback):
    def __init__(self):
        super().__init__()
        self.best_val_acc = 0
        
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        
        val_acc = logs.get('val_accuracy', 0)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            wandb.run.summary["best_val_accuracy"] = val_acc
            wandb.run.summary["best_epoch"] = epoch
```

## Troubleshooting

### Common Issues

1. **Summary not updating**: Ensure you're calling `run.summary.update()` when using the API
2. **Missing summary values**: Check that metrics are logged before run finishes
3. **Incorrect summary type**: Verify `define_metric` is called before logging begins

### Debug Tips

```python
# Check current summary values
print(run.summary._json_dict)

# Verify summary configuration
for metric in run._metrics:
    print(f"{metric}: {run._metrics[metric].get('summary', 'last')}")
```

## Conclusion

W&B Summary metrics are essential for experiment tracking and model comparison. They provide a clean interface for capturing key results, enabling efficient experimentation and analysis. By following these best practices and examples, you can leverage summary metrics to build better ML workflows and make data-driven decisions about your models.

## Further Resources

- [W&B Documentation: Log Summary Metrics](https://docs.wandb.ai/guides/track/log/log-summary)
- [W&B API Reference](https://docs.wandb.ai/ref/python/run#summary)
- [W&B Support: Log vs Summary](https://docs.wandb.ai/support/difference_log_summary)