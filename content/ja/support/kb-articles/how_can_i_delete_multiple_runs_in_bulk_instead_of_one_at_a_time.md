---
title: 複数の run を一度に削除するにはどうすればいいですか？
menu:
  support:
    identifier: >-
      ja-support-kb-articles-how_can_i_delete_multiple_runs_in_bulk_instead_of_one_at_a_time
support:
  - projects
  - runs
toc_hide: true
type: docs
url: /ja/support/:filename
translationKey: how_can_i_delete_multiple_runs_in_bulk_instead_of_one_at_a_time
---
[public API]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}}) を使用して、単一の操作で複数の run を削除します。

```python
import wandb

api = wandb.Api()
runs = api.runs('<entity>/<project>')
for run in runs:
    if <condition>:
        run.delete()  # 条件が合えば run を削除
```