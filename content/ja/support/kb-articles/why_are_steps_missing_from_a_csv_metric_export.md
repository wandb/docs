---
title: なぜ CSV メトリックのエクスポートで step が抜けているのでしょうか？
menu:
  support:
    identifier: ja-support-kb-articles-why_are_steps_missing_from_a_csv_metric_export
support:
- 実験
- run
toc_hide: true
type: docs
url: /support/:filename
---

エクスポート制限により、run の全履歴をCSVでエクスポートしたり、`run.history` API を利用したりすることができない場合があります。完全な run 履歴にアクセスするには、Parquet 形式で run history artifact をダウンロードしてください。

```python
import wandb
import pandas as pd

run = wandb.init()
artifact = run.use_artifact('<entity>/<project>/<run-id>-history:v0', type='wandb-history')
artifact_dir = artifact.download()
df = pd.read_parquet('<path to .parquet file>')
```