---
title: sweep で最良の run から Artifacts を見つけるにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-find_artifact_best_run_sweep
support:
- アーティファクト
toc_hide: true
type: docs
url: /ja/support/:filename
---

アーティファクトを sweep の中で最も良いパフォーマンスを示した run から取得するには、次のコードを使用します。

```python
api = wandb.Api()
sweep = api.sweep("entity/project/sweep_id")
runs = sorted(sweep.runs, key=lambda run: run.summary.get("val_acc", 0), reverse=True)
best_run = runs[0]
for artifact in best_run.logged_artifacts():
    artifact_path = artifact.download()
    print(artifact_path)
```
