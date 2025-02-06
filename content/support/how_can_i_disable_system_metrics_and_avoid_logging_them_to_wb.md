---
title: How can I disable logging of system metrics to W&B?
toc_hide: true
type: docs
tags:
  - metrics
  - runs
---

To disable logging of [system metrics]({{< relref "/guides/models/app/settings-page/system-metrics.md" >}}), set `_disable_stats` to `True`:

```python
wandb.init(settings=wandb.Settings(_disable_stats=True))
```