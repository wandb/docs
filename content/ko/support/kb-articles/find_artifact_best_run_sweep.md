---
title: How do I find an artifact from the best run in a sweep?
menu:
  support:
    identifier: ko-support-kb-articles-find_artifact_best_run_sweep
support:
- artifacts
toc_hide: true
type: docs
url: /ko/support/:filename
translationKey: find_artifact_best_run_sweep
---
스윕에서 가장 성능이 좋은 run에서 아티팩트를 검색하려면 다음 코드를 사용하세요.

```python
api = wandb.Api()
sweep = api.sweep("entity/project/sweep_id")
runs = sorted(sweep.runs, key=lambda run: run.summary.get("val_acc", 0), reverse=True)
best_run = runs[0]
for artifact in best_run.logged_artifacts():
    artifact_path = artifact.download()
    print(artifact_path)
```
