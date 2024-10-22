---
title: "How can I organize my logged charts and media in the W&B UI?"
tags:
   - experiments
---
The `/` character separates logged panels in the W&B UI. By default, the segment of the logged item's name before the `/` defines a group of panels known as a "Panel Section."

```python
wandb.log({"val/loss": 1.1, "val/acc": 0.3})
wandb.log({"train/loss": 0.1, "train/acc": 0.94})
```

In the [Workspace](../guides/app/pages/workspaces.md) settings, adjust the grouping of panels based on either the first segment or all segments separated by `/`.