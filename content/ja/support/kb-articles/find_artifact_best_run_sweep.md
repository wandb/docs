---
title: How do I find an artifact from the best run in a sweep?
menu:
  support:
    identifier: ja-support-kb-articles-find_artifact_best_run_sweep
support:
- artifacts
toc_hide: true
type: docs
url: /support/:filename
---

Sweep で最高のパフォーマンスを発揮した run から Artifacts を取得するには、次のコードを使用します。

```python
api = wandb.Api()
sweep = api.sweep("entity/project/sweep_id")
runs = sorted(sweep.runs, key=lambda run: run.summary.get("val_acc", 0), reverse=True)
best_run = runs[0]
for artifact in best_run.logged_artifacts():
    artifact_path = artifact.download()
    print(artifact_path)
```
