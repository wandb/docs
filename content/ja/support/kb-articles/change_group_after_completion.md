---
title: Is it possible to change the group assigned to a run after completion?
menu:
  support:
    identifier: ja-support-kb-articles-change_group_after_completion
support:
- runs
toc_hide: true
type: docs
url: /support/:filename
---

API を使用して、完了した run に割り当てられたグループを変更できます。この機能は Web UI には表示されません。グループを更新するには、次のコードを使用します。

```python
import wandb

api = wandb.Api()
run = api.run("<ENTITY>/<PROJECT>/<RUN_ID>")
run.group = "NEW-GROUP-NAME"
run.update()
```
