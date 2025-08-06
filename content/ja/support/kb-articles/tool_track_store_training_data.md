---
title: あなたのツールはトレーニングデータを追跡または保存しますか？
url: /support/:filename
toc_hide: true
type: docs
support:
- 実験
---

SHA もしくは一意の識別子を `wandb.Run.config.update(...)` に渡すことで、データセットをトレーニング run に関連付けることができます。W&B は、ローカルファイル名を指定して `wandb.Run.save()` が呼ばれない限り、データを保存しません。