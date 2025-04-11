---
title: 完了後に run に割り当てられたグループを変更することは可能ですか？
menu:
  support:
    identifier: ja-support-kb-articles-change_group_after_completion
support:
- runs
toc_hide: true
type: docs
url: /support/:filename
---

You can change the group assigned to a completed run using the API. This feature does not appear in the web UI. Use the following code to update the group:

```python
import wandb

# APIを使用して group を変更する
api = wandb.Api()
run = api.run("<ENTITY>/<PROJECT>/<RUN_ID>")
run.group = "NEW-GROUP-NAME"
run.update()
```