---
title: CSV 메트릭 내보내기에서 일부 step 이 누락되는 이유는 무엇인가요?
menu:
  support:
    identifier: ko-support-kb-articles-why_are_steps_missing_from_a_csv_metric_export
support:
- Experiments
- run
toc_hide: true
type: docs
url: /support/:filename
---

Export 제한으로 인해 전체 run 이력을 CSV로 내보내거나 `run.history` API를 사용할 수 없는 경우가 있습니다. 전체 run 이력에 엑세스하려면 Parquet 포맷을 사용하여 run history 아티팩트를 다운로드하세요:

```python
import wandb
import pandas as pd

run = wandb.init()
artifact = run.use_artifact('<entity>/<project>/<run-id>-history:v0', type='wandb-history')
artifact_dir = artifact.download()
df = pd.read_parquet('<path to .parquet file>')
```