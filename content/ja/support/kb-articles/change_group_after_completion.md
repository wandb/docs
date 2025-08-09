---
title: 完了した run に割り当てられているグループを後から変更することはできますか？
menu:
  support:
    identifier: ja-support-kb-articles-change_group_after_completion
support:
- runs
toc_hide: true
type: docs
url: /support/:filename
---

API を使って完了した run のグループを変更できます。この機能は Web UI には表示されません。グループを更新するには、以下のコードを使用してください。

```python
import wandb

api = wandb.Api()
run = api.run("<ENTITY>/<PROJECT>/<RUN_ID>")
# グループ名を新しいものに変更
run.group = "NEW-GROUP-NAME"
run.update()
```