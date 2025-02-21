---
title: Why are steps missing from a CSV metric export?
menu:
  support:
    identifier: ja-support-why_are_steps_missing_from_a_csv_metric_export
tags:
- experiments
- runs
toc_hide: true
type: docs
---

エクスポートの制限により、run の履歴全体を CSV としてエクスポートしたり、`run.history` API を使用したりできなくなる場合があります。run の履歴全体にアクセスするには、Parquet 形式で run の履歴 Artifact をダウンロードします。

```python
import wandb
import pandas as pd

run = wandb.init()
artifact = run.use_artifact('<entity>/<project>/<run-id>-history:v0', type='wandb-history') # Artifact を使用して、run の履歴にアクセスします。
artifact_dir = artifact.download() # Artifact をダウンロードします。
df = pd.read_parquet('<path to .parquet file>') # ダウンロードした Artifact から pandas DataFrame を作成します。
```
