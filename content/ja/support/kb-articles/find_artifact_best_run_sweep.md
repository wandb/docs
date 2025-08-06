---
title: sweep で最も良い run からアーティファクトを見つけるにはどうすればよいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- アーティファクト
---

最高のパフォーマンスを示した run から Artifacts を取得するには、以下のコードを使用します。

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