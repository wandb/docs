---
title: How can I reduce how frequently to log system metrics?
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