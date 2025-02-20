---
menu:
  support:
    identifier: ko-support-how_can_i_disable_logging_of_system_metrics_to_wb
tags:
- metrics
- runs
title: How can I disable logging of system metrics to W&B?
toc_hide: true
type: docs
---

To disable logging of [system metrics]({{< relref path="/guides/models/app/settings-page/system-metrics.md" lang="ko" >}}), set `_disable_stats` to `True`:

```python
wandb.init(settings=wandb.Settings(x_disable_stats=True))
```