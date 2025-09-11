---
title: Data Types
module: wandb.sdk.data_types
weight: 2
no_list: true
---

Data Types in W&B are classes that wrap media and structured data for logging to runs. They include visualization components in the W&B UI and handle data serialization, storage, and retrieval.

## Available Data Types

| Data Type | Description |
|-----------|-------------|
| [`Image`](./Image/) | Log images with support for masks, bounding boxes, and segmentation. |
| [`Video`](./Video/) | Track video data for model outputs or dataset samples. |
| [`Audio`](./Audio/) | Log audio samples for audio processing tasks. |
| [`Table`](./Table/) | Create tables that can contain mixed media types. |
| [`Plotly`](./Plotly/) | Log Plotly charts for data visualization. |
| [`Html`](./Html/) | Embed custom HTML content. |
| [`Object3D`](./Object3D/) | Visualize 3D point clouds and meshes. |
| [`Molecule`](./Molecule/) | Log molecular structures for computational chemistry. |
| [`Box3D`](./box3d/) | Track 3D bounding boxes for 3D object detection. |

## Examples

This example uses an `Image`:

```python
import wandb
import matplotlib.pyplot as plt

# Generate an image for demonstration purposes
path_to_img = "/path/to/cafe.png"
im = plt.imread(path_to_img)

# Initialize a new run
with wandb.init(project="awesome-project") as run:

    # Log the image
    run.log({"img": [wandb.Image(im, caption="Cafe")]})
```

This example uses a `Table` to log a table with mixed text and labels:

```python
import wandb

# Initialize a new run
with wandb.init(project="visualize-predictions", name="tables") as run:

    # Create tabular data, using a list of lists
    data = [["Cat", "1", "1"],["Dog", "0", "-1"]]
    run.log({"Table 1": wandb.Table(data=data, columns=["Text", "Predicted Label", "True Label"])})

    # Create tabular data, using `wandb.Table.add_data()` method
    table = wandb.Table(columns=["Text", "Predicted Label", "True Label"])
    table.add_data("Cat", "1", "1")
    table.add_data("Dog", "0", "-1")
    run.log({"Table 2": table})
```