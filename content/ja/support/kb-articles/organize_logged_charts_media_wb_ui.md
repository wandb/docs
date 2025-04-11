---
title: W&B UI でログしたチャートやメディアをどのように整理できますか？
menu:
  support:
    identifier: ja-support-kb-articles-organize_logged_charts_media_wb_ui
support:
- experiments
toc_hide: true
type: docs
url: /support/:filename
---

`/` 文字は W&B UI でログされたパネルを区切ります。デフォルトでは、`/` の前のログされたアイテムの名前のセグメントが、「パネルセクション」として知られるパネルのグループを定義します。

```python
wandb.log({"val/loss": 1.1, "val/acc": 0.3})
wandb.log({"train/loss": 0.1, "train/acc": 0.94})
```

[Workspace]({{< relref path="/guides/models/track/project-page.md#workspace-tab" lang="ja" >}}) 設定で、`/` で区切られた最初のセグメントまたはすべてのセグメントに基づいてパネルのグループ化を調整します。