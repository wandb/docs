---
title: Experiments
module: wandb
weight: 2
no_list: true
---

These classes comprise the core building blocks for tracking machine learning experiments, managing artifacts, and configuring SDK behavior. These foundational classes enable you to log metrics, store model checkpoints, version datasets, and manage experiment configurations with full reproducibility and collaboration features. 

> For more details on using these classes in ML experiments, consult the [Experiments]({{< relref "/guides/models/track/" >}}) and [Artifacts]({{< relref "/guides/core/artifacts/" >}}) docs.

## Core Classes

| Class | Description |
|-------|-------------|
| [`Run`](./run/) | The primary unit of computation logged by W&B, representing a single ML experiment with metrics, configurations, and outputs. |
| [`Artifact`](./artifact/) | Flexible and lightweight building block for dataset and model versioning with automatic deduplication and lineage tracking. |
| [`Settings`](./settings/) | Configuration management for the W&B SDK, controlling behavior from logging to API interactions. |

## Getting Started

### Track an experiment

Create and track a machine learning experiment with metrics logging:

```python
import wandb

# Initialize a new run
with wandb.init(project="my-experiments", config={"learning_rate": 0.001}) as run:
    # Access configuration
    config = run.config
    
    # Log metrics during training
    for epoch in range(10):
        metrics = train_one_epoch()  # Your training logic
        run.log({
            "loss": metrics["loss"],
            "accuracy": metrics["accuracy"],
            "epoch": epoch
        })
    
    # Log summary metrics
    run.summary["best_accuracy"] = max_accuracy
```

### Version a model artifact

Create and log a versioned model artifact with metadata:

```python
import wandb

with wandb.init(project="my-models") as run:
    # Train your model
    model = train_model()
    
    # Create an artifact for the model
    model_artifact = wandb.Artifact(
        name="my-model",
        type="model",
        description="ResNet-50 trained on ImageNet subset",
        metadata={
            "architecture": "ResNet-50",
            "dataset": "ImageNet-1K",
            "accuracy": 0.95
        }
    )
    
    # Add model files to the artifact
    model_artifact.add_file("model.pt")
    model_artifact.add_dir("model_configs/")
    
    # Log the artifact to W&B
    run.log_artifact(model_artifact)
```

### Configure SDK settings

Customize W&B SDK behavior for your specific requirements:

```python
import wandb

# Configure settings programmatically
wandb.Settings(
    project="production-runs",
    entity="my-team",
    mode="offline",  # Run offline, sync later
    save_code=True,   # Save source code
    quiet=True        # Reduce console output
)

# Or use environment variables
# export WANDB_PROJECT=production-runs
# export WANDB_MODE=offline

# Initialize with custom settings
with wandb.init() as run:
    # Your experiment code here
    pass
```

### Link artifacts for lineage tracking

Track relationships between datasets, models, and evaluations:

```python
import wandb

with wandb.init(project="ml-pipeline") as run:
    # Use a dataset artifact
    dataset = run.use_artifact("dataset:v1")
    dataset_dir = dataset.download()
    
    # Train model using the dataset
    model = train_on_dataset(dataset_dir)
    
    # Create model artifact with dataset lineage
    model_artifact = wandb.Artifact(
        name="trained-model",
        type="model"
    )
    model_artifact.add_file("model.pt")
    
    # Log with automatic lineage tracking
    run.log_artifact(model_artifact)
```
