---
title: CSV メトリックエクスポートにステップが欠落しているのはなぜですか？
menu:
  support:
    identifier: ja-support-kb-articles-why_are_steps_missing_from_a_csv_metric_export
support:
- 実験管理
- runs
toc_hide: true
type: docs
url: /ja/support/:filename
---

エクスポート制限により、実行履歴全体をCSVとしてエクスポートしたり、`run.history` APIを使用したりすることができない場合があります。完全な実行履歴にアクセスするには、Parquet形式を使用して実行履歴アーティファクトをダウンロードしてください。

```python
import wandb
import pandas as pd

run = wandb.init()
artifact = run.use_artifact('<entity>/<project>/<run-id>-history:v0', type='wandb-history')
artifact_dir = artifact.download()
df = pd.read_parquet('<path to .parquet file>')
```