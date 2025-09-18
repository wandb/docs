---
url: /support/:filename
title: How can I disable logging of system metrics to W&B?
toc_hide: true
type: docs
support:
  - metrics
  - runs
---

To disable logging of [system metrics]({{< relref "/ref/python/experiments/system-metrics.md" >}}), set `_disable_stats` to `True`:

```python
wandb.init(settings=wandb.Settings(x_disable_stats=True))
```