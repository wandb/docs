---
title: Does your tool track or store training data?
menu:
  support:
    identifier: ja-support-kb-articles-tool_track_store_training_data
support:
- experiments
toc_hide: true
type: docs
url: /support/:filename
---

SHAまたは固有の識別子を `wandb.config.update(...)` に渡して、データセットをトレーニング run に関連付けます。W&B は、ローカルファイル名で `wandb.save` が呼び出されない限り、データを保存しません。
