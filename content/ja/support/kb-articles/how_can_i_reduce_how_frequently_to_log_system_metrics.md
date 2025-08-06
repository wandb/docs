---
title: システムメトリクスのログ頻度を変更するにはどうすればよいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- メトリクス
- run
---

[システムメトリクス]({{< relref "/guides/models/app/settings-page/system-metrics.md" >}}) をログする頻度を設定するには、`_stats_sampling_interval` を秒数（float 型）で指定します。デフォルトは `10.0` です。

```python
wandb.init(settings=wandb.Settings(x_stats_sampling_interval=30.0))
```