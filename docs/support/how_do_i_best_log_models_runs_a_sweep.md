---
title: "How do I best log models from runs in a sweep?"
tags:
   - artifacts
---

One effective pattern for logging models in a [sweep](../sweeps/intro.md) is to have a model artifact for the sweep, where the versions will correspond to different runs from the sweep. More concretely, you would have:

```python
wandb.Artifact(name="sweep_name", type="model")
```