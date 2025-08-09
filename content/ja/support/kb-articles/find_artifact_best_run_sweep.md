---
title: sweep の中で最も良い run からアーティファクトを見つけるにはどうすればいいですか？
menu:
  support:
    identifier: ja-support-kb-articles-find_artifact_best_run_sweep
support:
- アーティファクト
toc_hide: true
type: docs
url: /support/:filename
---

スイープ内で最も良いパフォーマンスを出した run からアーティファクトを取得するには、以下のコードを使います。

```python
api = wandb.Api()
sweep = api.sweep("entity/project/sweep_id")
runs = sorted(sweep.runs, key=lambda run: run.summary.get("val_acc", 0), reverse=True)
best_run = runs[0]
for artifact in best_run.logged_artifacts():
    # アーティファクトをダウンロード
    artifact_path = artifact.download()
    print(artifact_path)
```