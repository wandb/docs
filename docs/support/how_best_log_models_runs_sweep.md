---
title: "How do I best log models from runs in a sweep?"
displayed_sidebar: support
tags:
   - artifacts
---
One effective approach for logging models in a [sweep](../guides/sweeps/intro.md) involves creating a model artifact for the sweep. Each version represents a different run from the sweep. Implement it as follows:

```python
wandb.Artifact(name="sweep_name", type="model")
```