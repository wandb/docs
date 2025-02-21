---
title: How do I find an artifact from the best run in a sweep?
menu:
  support:
    identifier: ja-support-find_artifact_best_run_sweep
tags:
- artifacts
toc_hide: true
type: docs
---

最高のパフォーマンスを発揮した run からアーティファクトを取得するには、次のコードを使用します。

```python
api = wandb.Api()
sweep = api.sweep("entity/project/sweep_id")
runs = sorted(sweep.runs, key=lambda run: run.summary.get("val_acc", 0), reverse=True)
best_run = runs[0]
for artifact in best_run.logged_artifacts():
    artifact_path = artifact.download()
    print(artifact_path)
```