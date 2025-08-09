---
title: 스윕에서 가장 성능이 좋은 run의 아티팩트를 어떻게 찾을 수 있나요?
menu:
  support:
    identifier: ko-support-kb-articles-find_artifact_best_run_sweep
support:
- 아티팩트
toc_hide: true
type: docs
url: /support/:filename
---

스윕에서 가장 성능이 좋은 run 의 아티팩트를 가져오려면 다음 코드를 사용하세요.

```python
api = wandb.Api()
sweep = api.sweep("entity/project/sweep_id")
runs = sorted(sweep.runs, key=lambda run: run.summary.get("val_acc", 0), reverse=True)
best_run = runs[0]
for artifact in best_run.logged_artifacts():
    artifact_path = artifact.download()
    print(artifact_path)
```