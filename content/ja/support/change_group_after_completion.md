---
title: Is it possible to change the group assigned to a run after completion?
menu:
  support:
    identifier: ja-support-change_group_after_completion
tags:
- runs
toc_hide: true
type: docs
---

完了した run に割り当てられたグループを API を使用して変更することができます。この機能はウェブ UI には表示されません。次のコードを使用してグループを更新してください:

```python
import wandb

api = wandb.Api()
run = api.run("<ENTITY>/<PROJECT>/<RUN_ID>")
run.group = "NEW-GROUP-NAME"  # 新しいグループ名
run.update()
```