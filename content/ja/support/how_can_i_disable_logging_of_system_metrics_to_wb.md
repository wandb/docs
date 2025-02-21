---
title: How can I disable logging of system metrics to W&B?
menu:
  support:
    identifier: ja-support-how_can_i_disable_logging_of_system_metrics_to_wb
tags:
- metrics
- runs
toc_hide: true
type: docs
---

[system metrics]({{< relref path="/guides/models/app/settings-page/system-metrics.md" lang="ja" >}}) のロギングを無効にするには、`_disable_stats` を `True` に設定します。

```python
wandb.init(settings=wandb.Settings(x_disable_stats=True))
```