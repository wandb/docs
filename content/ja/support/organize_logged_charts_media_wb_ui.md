---
title: How can I organize my logged charts and media in the W&B UI?
menu:
  support:
    identifier: ja-support-organize_logged_charts_media_wb_ui
tags:
- experiments
toc_hide: true
type: docs
---

`/` キャラクターは、W&B UI でログされたパネルを区切ります。デフォルトでは、`/` の前のログされた項目の名前のセグメントが「パネルセクション」として知られるパネルのグループを定義します。

```python
wandb.log({"val/loss": 1.1, "val/acc": 0.3})
wandb.log({"train/loss": 0.1, "train/acc": 0.94})
```

[Workspace]({{< relref path="/guides/models/track/project-page.md#workspace-tab" lang="ja" >}}) 設定で、`/` で区切られた最初のセグメントまたはすべてのセグメントに基づいて、パネルのグループ化を調整します。