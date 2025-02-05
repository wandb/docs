---
title: Why does exporting a metric as CSV not download all the steps?
toc_hide: true
type: docs
tags:
  - experiments
  - runs
---

There are some export limits, so you may not be able to export the entire run history via CSV or the run.history API. To access the complete run history, download the run history artifact parquet file:

```python
import wandb
import pandas as pd

run = wandb.init()
artifact = run.use_artifact('<entity>/<project>/<run-id>-history:v0', type='wandb-history')
artifact_dir = artifact.download()
df = pd.read_parquet('<path to .parquet file>')
```
