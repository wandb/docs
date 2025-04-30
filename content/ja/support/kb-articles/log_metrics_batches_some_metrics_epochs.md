---
title: エポックごとだけメトリクスをログし、バッチごとのメトリクスのログを避けたい場合はどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-log_metrics_batches_some_metrics_epochs
support:
  - experiments
  - metrics
toc_hide: true
type: docs
url: /ja/support/:filename
---
各バッチで特定のメトリクスをログし、プロットを標準化するために、希望する x 軸の値とメトリクスを一緒にログします。カスタムプロットで編集をクリックし、カスタム x 軸を選択してください。

```python
wandb.log({"batch": batch_idx, "loss": 0.3})
wandb.log({"epoch": epoch, "val_acc": 0.94})
```