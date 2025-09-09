---
title: 複数の Runs を一括で削除するにはどうすればよいですか？
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

複数の run を 1 回の操作で削除するには、[公開 API]({{< relref path="/ref/python/public-api/api.md" lang="ja" >}}) を使用します:

```python
import wandb

api = wandb.Api()
runs = api.runs('<entity>/<project>')
for run in runs:
    if <condition>:
        run.delete()
```