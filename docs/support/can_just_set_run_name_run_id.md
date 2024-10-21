---
title: "Can I just set the run name to the run ID?"
tags:
   - experiments
---

If you'd like to overwrite the run name (like snowy-owl-10) with the run ID (like qvlp96vk) you can use this snippet:

```python
import wandb

wandb.init()
wandb.run.name = wandb.run.id
wandb.run.save()
```