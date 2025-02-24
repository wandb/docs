---
title: Why are steps missing from a CSV metric export?
menu:
  support:
    identifier: ko-support-why_are_steps_missing_from_a_csv_metric_export
tags:
- experiments
- runs
toc_hide: true
type: docs
---

내보내기 제한으로 인해 전체 run 기록을 CSV로 내보내거나 `run.history` API를 사용하는 것이 불가능할 수 있습니다. 전체 run 기록에 엑세스하려면 Parquet 형식을 사용하여 run 기록 아티팩트를 다운로드하세요.

```python
import wandb
import pandas as pd

run = wandb.init()
artifact = run.use_artifact('<entity>/<project>/<run-id>-history:v0', type='wandb-history')
artifact_dir = artifact.download()
df = pd.read_parquet('<path to .parquet file>')
```
