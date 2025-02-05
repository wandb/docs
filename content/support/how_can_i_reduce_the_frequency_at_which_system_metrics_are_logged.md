---
title: How can I reduce the frequency at which System Metrics are logged?
toc_hide: true
type: docs
tags:
  - metrics
  - runs
---

Adjust the sampling interval (default is 10s):

```python
wandb.init(settings=wandb.Settings(_stats_sampling_interval=30.0))
```