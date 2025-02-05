---
title: How can I disable System Metrics and avoid logging them to W&B?
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