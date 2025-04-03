---
title: How can I organize my logged charts and media in the W&B UI?
menu:
  support:
    identifier: ja-support-kb-articles-organize_logged_charts_media_wb_ui
support:
- experiments
toc_hide: true
type: docs
url: /support/:filename
---

`/` 文字は、W&B UI でログに記録された パネル を区切ります。デフォルトでは、ログに記録された項目の名前の `/` より前のセグメントは、「パネル セクション」と呼ばれる パネル のグループを定義します。

```python
wandb.log({"val/loss": 1.1, "val/acc": 0.3})
wandb.log({"train/loss": 0.1, "train/acc": 0.94})
```

[Workspace]({{< relref path="/guides/models/track/project-page.md#workspace-tab" lang="ja" >}}) の 設定 で、`/` で区切られた最初のセグメントまたはすべてのセグメントに基づいて、 パネル のグループ化を調整します。
