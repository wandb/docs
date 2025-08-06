---
title: run 完了後に割り当てられたグループを変更することはできますか？
url: /support/:filename
toc_hide: true
type: docs
support:
- run
---

API を使って完了した run に割り当てられているグループを変更できます。この機能は Web UI には表示されません。グループを更新するには、以下のコードを使用してください。

```python
import wandb

api = wandb.Api()
run = api.run("<ENTITY>/<PROJECT>/<RUN_ID>")
run.group = "NEW-GROUP-NAME"  # 新しいグループ名を設定
run.update()  # 変更を保存
```