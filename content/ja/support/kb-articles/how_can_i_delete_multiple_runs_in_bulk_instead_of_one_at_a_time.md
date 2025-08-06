---
title: 複数の run をまとめて一度に削除するにはどうすればいいですか？
menu:
  support:
    identifier: ja-support-kb-articles-how_can_i_delete_multiple_runs_in_bulk_instead_of_one_at_a_time
support:
- プロジェクト
- runs
toc_hide: true
type: docs
url: /support/:filename
---

[パブリック API]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}}) を使って、複数の run を一度に削除できます。

```python
import wandb

api = wandb.Api()
runs = api.runs('<entity>/<project>')
for run in runs:
    if <condition>:  # 条件を指定
        run.delete()  # run を削除
```