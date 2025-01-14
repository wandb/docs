---
title: Data Types
---


{{< cta-button githubLink="https://www.github.com/wandb/wandb/tree/v0.19.2/wandb/__init__.py" >}}

This module defines data types for logging rich, interactive visualizations to W&B.

Data types include common media types, like images, audio, and videos,
flexible containers for information, like tables and HTML, and more.

For more on logging media, see [our guide](https://docs.wandb.com/guides/track/log/media)

For more on logging structured data for interactive dataset and model analysis,
see [our guide to W&B Tables](https://docs.wandb.com/guides/tables/).

All of these special data types are subclasses of WBValue. All the data types
serialize to JSON, since that is what wandb uses to save the objects locally
and upload them to the W&B server.

## Classes

[`class Audio`](./audio/): Wandb class for audio clips.

[`class BoundingBoxes2D`](./boundingboxes2d/): Format images with 2D bounding box overlays for logging to W&B.

[`class Graph`](./graph/): Wandb class for graphs.

[`class Histogram`](./histogram/): wandb class for histograms.

[`class Html`](./html/): Wandb class for arbitrary html.

[`class Image`](./image/): Format images for logging to W&B.

[`class ImageMask`](./imagemask/): Format image masks or overlays for logging to W&B.

[`class Molecule`](./molecule/): Wandb class for 3D Molecular data.

[`class Object3D`](./object3d/): Wandb class for 3D point clouds.

[`class Plotly`](./plotly/): Wandb class for plotly plots.

[`class Table`](./table/): The Table class used to display and analyze tabular data.

[`class Video`](./video/): Format a video for logging to W&B.

[`class WBTraceTree`](./wbtracetree/): Media object for trace tree data.
