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

全ての run の履歴を CSV としてエクスポートすることや、`run.history` API を使用することをエクスポート制限が妨げる場合があります。完全な run 履歴にアクセスするには、Parquet 形式を使用して run 履歴アーティファクトをダウンロードしてください。

```python
import wandb
import pandas as pd

run = wandb.init()
artifact = run.use_artifact('<entity>/<project>/<run-id>-history:v0', type='wandb-history')
artifact_dir = artifact.download()
df = pd.read_parquet('<path to .parquet file>')
```