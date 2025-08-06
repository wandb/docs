---
title: W&B の UI でログしたチャートやメディアを整理するにはどうすればいいですか？
url: /support/:filename
toc_hide: true
type: docs
support:
- 実験
---

「/」文字は、W&B UI でログされたパネルを区切ります。デフォルトでは、ログ項目の名前の「/」の前のセグメントが、「パネルセクション」と呼ばれるパネルのグループを定義します。

```python
import wandb

with wandb.init() as run:

   # 検証の損失と精度をログ
   run.log({"val/loss": 1.1, "val/acc": 0.3})
   # 学習の損失と精度をログ
   run.log({"train/loss": 0.1, "train/acc": 0.94})
```

[Workspace]({{< relref "/guides/models/track/project-page.md#workspace-tab" >}}) の設定で、パネルのグループ分けを「/」で区切られた最初のセグメント、またはすべてのセグメントに基づいて調整できます。