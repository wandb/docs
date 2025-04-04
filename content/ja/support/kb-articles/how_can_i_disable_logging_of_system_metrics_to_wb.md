---
menu:
  support:
    identifier: ja-support-kb-articles-how_can_i_disable_logging_of_system_metrics_to_wb
support:
- metrics
- runs
title: How can I disable logging of system metrics to W&B?
toc_hide: true
type: docs
url: /support/:filename
---

To disable logging of [system metrics]({{< relref path="/guides/models/app/settings-page/system-metrics.md" lang="ja" >}}), set `_disable_stats` to `True`:

```python
wandb.init(settings=wandb.Settings(x_disable_stats=True))
```