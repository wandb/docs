---
title: "How do I best log models from runs in a sweep?"
toc_hide: true
type: docs
tags:
   - artifacts
   - sweeps
---
One effective approach for logging models in a [sweep]({{< relref "/guides/models/sweeps/" >}}) involves creating a model artifact for the sweep. Each version represents a different run from the sweep. Implement it as follows:

```python
wandb.Artifact(name="sweep_name", type="model")
```