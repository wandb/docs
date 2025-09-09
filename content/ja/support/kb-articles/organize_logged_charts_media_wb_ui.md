---
title: W&B の UI で、ログしたチャートやメディアをどのように整理できますか？
menu:
  support:
    identifier: ja-support-kb-articles-organize_logged_charts_media_wb_ui
support:
- 実験管理
toc_hide: true
type: docs
url: /support/:filename
---

`/` 文字は、W&B の UI でログしたパネルを区切ります。デフォルトでは、ログした項目名の `/' より前のセグメントが、"Panel Section" と呼ばれるパネルのグループを定義します。

```python
import wandb

with wandb.init() as run:

   run.log({"val/loss": 1.1, "val/acc": 0.3})
   run.log({"train/loss": 0.1, "train/acc": 0.94})
```

[Workspace]({{< relref path="/guides/models/track/project-page.md#workspace-tab" lang="ja" >}}) の 設定で、`/` で区切られた最初のセグメント、またはすべてのセグメントを基準にパネルのグルーピングを調整できます。