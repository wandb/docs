---
title: "How can I organize my logged charts and media in the W&B UI?"
tags:
   - experiments
---

We treat `/` as a separator for organizing logged panels in the W&B UI. By default, the component of the logged item's name before a `/` is used to define a group of panel called a "Panel Section".

```python
wandb.log({"val/loss": 1.1, "val/acc": 0.3})
wandb.log({"train/loss": 0.1, "train/acc": 0.94})
```

In the [Workspace](../../app/pages/workspaces.md) settings, you can change whether panels are grouped by just the first component or by all components separated by `/`.