---
title: バッチごとに記録したいメトリクスもあれば、エポックごとだけに記録したいメトリクスがある場合はどうすればいいですか？
menu:
  support:
    identifier: ja-support-kb-articles-log_metrics_batches_some_metrics_epochs
support:
- 実験
- メトリクス
toc_hide: true
type: docs
url: /support/:filename
---

各バッチで特定のメトリクスをログし、プロットを標準化するには、メトリクスと一緒に希望する x 軸の値もログしてください。カスタムプロットでは、編集ボタンをクリックしてカスタムの x 軸を選択できます。

```python
import wandb

with wandb.init() as run:
    # バッチごとのメトリクスと x 軸の値をログする
    run.log({"batch": batch_idx, "loss": 0.3})
    # エポックごとのメトリクスをログする
    run.log({"epoch": epoch, "val_acc": 0.94})
```
