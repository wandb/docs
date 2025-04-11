---
title: run 完了後に割り当てられたグループを変更することは可能ですか？
menu:
  support:
    identifier: ja-support-kb-articles-change_group_after_completion
support:
- run
toc_hide: true
type: docs
url: /support/:filename
---

完了した run に割り当てられたグループを API を使用して変更することができます。この機能は Web UI には表示されません。次のコードを使用してグループを更新してください：

```python
import wandb

# APIを使用して group を変更する
api = wandb.Api()
run = api.run("<ENTITY>/<PROJECT>/<RUN_ID>")
run.group = "NEW-GROUP-NAME"
run.update()
```