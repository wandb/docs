---
title: Optional Extras
weight: 10
---

# W&B Optional Extras Reference

The `wandb` package provides optional extras that enable additional functionality for specific use cases. These extras install additional dependencies required for specialized features.

## Installation

Install extras using pip with square brackets notation:

```bash
# Single extra
pip install "wandb[media]"

# Multiple extras
pip install "wandb[media,sweeps,launch]"

# Note: In zsh (default on macOS), use quotes or escape brackets
pip install wandb\[media\]
```

## Available Extras

### `wandb[media]`

Enables advanced media logging capabilities for visualizations and multimedia content.

**Installed packages:**
- `bokeh` - Interactive visualization library
- `moviepy` - Video editing and processing
- `pillow` - Image processing (PIL fork)
- `plotly` - Interactive graphing library
- `imageio` - Image I/O operations
- `rdkit` - Cheminformatics and molecule visualization
- `soundfile` - Audio file I/O

**Enabled features:**
- Advanced plotting with `wandb.plot` functions
- Video logging with `wandb.Video`
- Audio logging with `wandb.Audio`
- Molecule visualization for chemistry/drug discovery
- Enhanced image processing capabilities

**Example usage:**
```python
import wandb
import plotly.graph_objects as go

# Requires wandb[media]
wandb.init()

# Log interactive Plotly figures
fig = go.Figure(data=go.Bar(x=['A', 'B', 'C'], y=[1, 3, 2]))
wandb.log({"plotly_chart": wandb.Plotly(fig)})

# Log videos
wandb.log({"video": wandb.Video("path/to/video.mp4")})
```

### `wandb[workspaces]`

Provides the Workspaces API for programmatically creating and managing W&B workspaces and reports.

**Installed packages:**
- `wandb-workspaces` - Official W&B workspaces library

**Documentation:**
See the dedicated [Workspaces API Reference]({{< relref "/ref/python/wandb_workspaces" >}})

**Example usage:**
```python
import wandb_workspaces.workspaces as ws

workspace = ws.Workspace(
    entity="team", 
    project="my-project",
    sections=[
        ws.Section(
            name="Metrics",
            panels=[
                ws.LinePlot(x="Step", y=["loss", "accuracy"])
            ]
        )
    ]
)
workspace.save()
```

### `wandb[sweeps]`

Enhanced support for hyperparameter sweeps and optimization.

**Enabled features:**
- Advanced sweep configuration options
- Additional sweep controllers
- Enhanced parameter sampling strategies

**Example usage:**
```python
import wandb

sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'loss', 'goal': 'minimize'},
    'parameters': {
        'learning_rate': {'min': 0.001, 'max': 0.1},
        'batch_size': {'values': [16, 32, 64]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="my-project")
wandb.agent(sweep_id, function=train_fn, count=10)
```

### `wandb[launch]`

Enables W&B Launch for job orchestration and deployment.

**Enabled features:**
- Job queueing and scheduling
- Resource management
- Deployment to various compute backends
- Job templates and configurations

**Example usage:**
```python
import wandb
from wandb.sdk.launch import launch

# Launch a job
launch(
    uri="https://github.com/my-org/my-repo",
    project="my-project",
    entity="my-team",
    config={"learning_rate": 0.01}
)
```

### `wandb[models]`

Enhanced model management and versioning capabilities.

**Enabled features:**
- Model registry operations
- Model lineage tracking
- Advanced model artifact handling
- Model performance comparison tools

**Example usage:**
```python
import wandb

run = wandb.init()

# Log model with enhanced metadata
model_artifact = wandb.Artifact(
    name="my-model",
    type="model",
    metadata={
        "framework": "pytorch",
        "architecture": "resnet50",
        "dataset": "imagenet"
    }
)
model_artifact.add_file("model.pth")
run.log_artifact(model_artifact)
```

### Cloud Provider Integrations

#### `wandb[aws]`

AWS integration for S3 artifact storage and SageMaker support.

**Enabled features:**
- Direct S3 artifact upload/download
- SageMaker training job integration
- AWS authentication helpers
- Optimized data transfer for AWS regions

#### `wandb[azure]`

Azure integration for blob storage and Azure ML support.

**Enabled features:**
- Azure Blob Storage for artifacts
- Azure ML workspace integration
- Azure authentication helpers
- Optimized data transfer for Azure regions

#### `wandb[gcp]`

Google Cloud Platform integration.

**Enabled features:**
- GCS (Google Cloud Storage) for artifacts
- Vertex AI integration
- GCP authentication helpers
- Optimized data transfer for GCP regions

### `wandb[kubeflow]`

Integration with Kubeflow pipelines and workflows.

**Enabled features:**
- Kubeflow pipeline component wrappers
- Automatic pipeline metadata logging
- Kubeflow artifact handling
- Pipeline visualization in W&B

**Example usage:**
```python
from wandb.integration.kubeflow import wandb_log

@wandb_log(project="my-project")
def training_component(learning_rate: float):
    # Your training code
    pass
```

### `wandb[importers]`

Tools for importing data from other experiment tracking systems.

**Enabled features:**
- TensorBoard log importer
- MLflow run importer
- CSV/JSON data importers
- Legacy format converters

**Example usage:**
```python
from wandb.apis.importers import import_tensorboard

import_tensorboard(
    log_dir="./tb_logs",
    project="imported-project",
    entity="my-team"
)
```

### `wandb[perf]`

Performance monitoring and profiling tools.

**Enabled features:**
- GPU memory profiling
- CPU/Memory profiling
- Training bottleneck detection
- Performance regression tracking

**Example usage:**
```python
import wandb
from wandb.profiler import profile

run = wandb.init()

with profile("training_step"):
    # Your training code
    model.train()
    loss.backward()
    optimizer.step()
```

## Checking Installed Extras

To verify which extras are installed:

```python
import wandb
import pkg_resources

# Check wandb version
print(f"wandb version: {wandb.__version__}")

# Check for specific extra dependencies
extras_to_check = {
    'media': ['plotly', 'bokeh', 'moviepy'],
    'workspaces': ['wandb_workspaces'],
    'sweeps': ['wandb'],  # Core functionality, check for version
}

for extra, packages in extras_to_check.items():
    print(f"\n{extra} extra:")
    for package in packages:
        try:
            version = pkg_resources.get_distribution(package).version
            print(f"  ✓ {package} {version}")
        except pkg_resources.DistributionNotFound:
            print(f"  ✗ {package} not installed")
```

## Best Practices

1. **Install only what you need**: Each extra brings additional dependencies. Install only the extras required for your use case.

2. **Version compatibility**: Some extras may have specific version requirements. Use a virtual environment to avoid conflicts.

3. **Production deployments**: For production, explicitly specify versions:
   ```bash
   pip install "wandb[media]==0.21.3"
   ```

4. **Docker images**: When building Docker images, install extras in a single layer:
   ```dockerfile
   RUN pip install "wandb[media,sweeps,launch]==0.21.3"
   ```

## Troubleshooting

### Import errors after installing extras

If you encounter import errors after installing extras:

1. Verify installation:
   ```bash
   pip list | grep wandb
   ```

2. Reinstall with forced upgrade:
   ```bash
   pip install --upgrade --force-reinstall "wandb[extra_name]"
   ```

3. Check for conflicting dependencies:
   ```bash
   pip check
   ```

### Shell escaping issues

If you get "no matches found" errors:

```bash
# Instead of:
pip install wandb[media]  # May fail in zsh

# Use:
pip install "wandb[media]"  # Works in all shells
# Or:
pip install wandb\[media\]  # Escaped brackets
```

## API Documentation Links

For detailed API documentation of features enabled by each extra:

- [Media and Visualization]({{< relref "/ref/python/sdk/data-types" >}}) - Data types for media logging
- [Sweeps]({{< relref "/guides/models/sweeps" >}}) - Hyperparameter optimization guide
- [Launch]({{< relref "/launch/" >}}) - Job orchestration documentation
- [Workspaces]({{< relref "/ref/python/wandb_workspaces" >}}) - Workspaces API reference
- [Integrations]({{< relref "/guides/integrations" >}}) - Integration guides

## Contributing

The extras are defined in the `wandb` package's `setup.py` or `pyproject.toml`. To request new extras or report issues:

1. Check the [W&B GitHub repository](https://github.com/wandb/wandb)
2. Open an issue with the "extras" label
3. For `wandb-workspaces`, see its [dedicated repository](https://github.com/wandb/wandb-workspaces)
