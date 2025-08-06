---
title: 複数の run を一括で削除するにはどうすればよいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- プロジェクト
- run
---

[パブリック API]({{< relref "/ref/python/public-api/api.md" >}}) を使って、複数の Run を一括で削除できます。

```python
import wandb

api = wandb.Api()
runs = api.runs('<entity>/<project>')
for run in runs:
    if <condition>:
        run.delete()
```
