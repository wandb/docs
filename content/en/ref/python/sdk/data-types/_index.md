---
title: Data Types
module: wandb.sdk.data_types
weight: 2
---

The W&B Python SDK includes data types for logging various forms of media and structured data. 

## Overview

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

## Getting Started

Example usage of W&B Data Types:

```python
import wandb
import numpy as np
from PIL import Image as PILImage

# Initialize a run
wandb.init(project="data-types-demo")

# Log an image with annotations
image = wandb.Image(
    PILImage.open("sample.jpg"),
    caption="Model prediction",
    boxes={
        "predictions": {
            "box_data": [{"position": {"minX": 10, "minY": 20, "maxX": 100, "maxY": 150},
                         "class_id": 1, "scores": {"confidence": 0.95}}]
        }
    }
)
wandb.log({"annotated_image": image})

# Create and log a table with mixed media
table = wandb.Table(columns=["id", "image", "prediction", "confidence"])
table.add_data("sample_1", wandb.Image("img1.jpg"), "cat", 0.95)
table.add_data("sample_2", wandb.Image("img2.jpg"), "dog", 0.87)
wandb.log({"results_table": table})

# Log 3D point cloud data
point_cloud = np.random.rand(1000, 3)  # 1000 points in 3D space
wandb.log({"point_cloud": wandb.Object3D(point_cloud)})

wandb.finish()
```

