---
title: sweep で最良の run からアーティファクトを見つけるにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-find_artifact_best_run_sweep
support:
- artifacts
toc_hide: true
type: docs
url: /support/:filename
---

sweep の中で最も良い結果を出した run からアーティファクトを取得するには、次のコードを使います:

```python
api = wandb.Api()
sweep = api.sweep("entity/project/sweep_id")
runs = sorted(sweep.runs, key=lambda run: run.summary.get("val_acc", 0), reverse=True)
best_run = runs[0]
for artifact in best_run.logged_artifacts():
    artifact_path = artifact.download()
    print(artifact_path)
```