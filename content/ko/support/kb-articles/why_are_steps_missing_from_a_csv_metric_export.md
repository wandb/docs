---
title: Why are steps missing from a CSV metric export?
menu:
  support:
    identifier: ko-support-kb-articles-why_are_steps_missing_from_a_csv_metric_export
support:
- experiments
- runs
toc_hide: true
type: docs
url: /ko/support/:filename
---

내보내기 제한으로 인해 전체 run 이력은 CSV로 내보내거나 `run.history` API를 사용하여 내보낼 수 없습니다. 전체 run 이력에 엑세스하려면 Parquet 형식을 사용하여 run 이력 아티팩트를 다운로드하세요.

```python
import wandb
import pandas as pd

run = wandb.init()
artifact = run.use_artifact('<entity>/<project>/<run-id>-history:v0', type='wandb-history')
artifact_dir = artifact.download()
df = pd.read_parquet('<path to .parquet file>')
```
