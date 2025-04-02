---
title: Why are steps missing from a CSV metric export?
menu:
  support:
    identifier: ja-support-kb-articles-why_are_steps_missing_from_a_csv_metric_export
support:
- experiments
- runs
toc_hide: true
type: docs
url: /support/:filename
---

エクスポートの制限により、run の履歴全体を CSV としてエクスポートしたり、`run.history` API を使用したりできなくなる場合があります。run の履歴全体にアクセスするには、Parquet 形式で run の履歴 Artifact をダウンロードします。

```python
import wandb
import pandas as pd

run = wandb.init()
artifact = run.use_artifact('<entity>/<project>/<run-id>-history:v0', type='wandb-history')
artifact_dir = artifact.download()
df = pd.read_parquet('<path to .parquet file>')
```