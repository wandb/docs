---
title: How can I disable logging of system metrics to W&B?
toc_hide: true
type: docs
tags:
  - metrics
  - runs
---

To disable System Metrics:

```python
wandb.init(settings=wandb.Settings(_disable_stats=True))
```