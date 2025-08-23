---
title: あなたのツールはトレーニングデータを追跡または保存しますか？
menu:
  support:
    identifier: ja-support-kb-articles-tool_track_store_training_data
support:
- 実験
toc_hide: true
type: docs
url: /support/:filename
---

SHA または一意の識別子を `wandb.Run.config.update(...)` に渡すことで、データセットをトレーニング run と関連付けることができます。W&B は、ローカルファイル名とともに `wandb.Run.save()` が呼ばれない限り、データは保存しません。