---
menu:
  support:
    identifier: ja-support-kb-articles-why_are_steps_missing_from_a_csv_metric_export
support:
- experiments
- runs
title: Why are steps missing from a CSV metric export?
toc_hide: true
type: docs
url: /support/:filename
---

Export limits can prevent the entire run history from being exported as a CSV or using the `run.history` API. To access the complete run history, download the run history artifact using Parquet format:

```python
import wandb
import pandas as pd

run = wandb.init()
artifact = run.use_artifact('<entity>/<project>/<run-id>-history:v0', type='wandb-history')
artifact_dir = artifact.download()
df = pd.read_parquet('<path to .parquet file>')
```