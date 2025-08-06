---
title: W&B へのシステムメトリクスのログを無効にするにはどうすればよいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- メトリクス
- run
---

[システムメトリクス]({{< relref "/guides/models/app/settings-page/system-metrics.md" >}}) のログを無効にするには、`_disable_stats` を `True` に設定します。

```python
# システムメトリクスのログを無効化
wandb.init(settings=wandb.Settings(x_disable_stats=True))
```