---
title: このツールはトレーニングデータを追跡または保存しますか？
menu:
  support:
    identifier: ja-support-kb-articles-tool_track_store_training_data
support:
- 実験
toc_hide: true
type: docs
url: /support/:filename
---

SHA または一意の識別子を `wandb.Run.config.update(...)` に渡して、データセットをトレーニング run に関連付けます。ローカルファイル名を指定して `wandb.Run.save()` が呼び出されない限り、W&B はデータを保存しません。