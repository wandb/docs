---
title: W&B UI でログしたチャートやメディアを整理するにはどうすればよいですか？
menu:
  support:
    identifier: ja-support-kb-articles-organize_logged_charts_media_wb_ui
support:
- 実験
toc_hide: true
type: docs
url: /support/:filename
---

「/」文字は、W&B UI でログされたパネルを区切るために使われます。デフォルトでは、ログされたアイテム名の「/」より前の部分が「パネルセクション」と呼ばれるパネルのグループを定義します。

```python
import wandb

with wandb.init() as run:

   run.log({"val/loss": 1.1, "val/acc": 0.3})
   run.log({"train/loss": 0.1, "train/acc": 0.94})
```

[Workspace]({{< relref path="/guides/models/track/project-page.md#workspace-tab" lang="ja" >}}) の設定では、パネルのグループ化を「/」で区切られた最初のセグメント、または全てのセグメントに基づいて調整できます。