---
title: CSV メトリクス エクスポートにステップが含まれていないのはなぜですか？
menu:
  support:
    identifier: ja-support-kb-articles-why_are_steps_missing_from_a_csv_metric_export
support:
- 実験
- runs
toc_hide: true
type: docs
url: /support/:filename
---

エクスポートの制限により、run の履歴全体を CSV として、または `run.history` API を使ってエクスポートできない場合があります。完全な run の履歴にアクセスするには、Parquet 形式で run 履歴のアーティファクトをダウンロードしてください:

```python
import wandb
import pandas as pd

run = wandb.init()
artifact = run.use_artifact('<entity>/<project>/<run-id>-history:v0', type='wandb-history')
artifact_dir = artifact.download()
df = pd.read_parquet('<path to .parquet file>')
```