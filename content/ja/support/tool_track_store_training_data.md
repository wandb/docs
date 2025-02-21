---
title: Does your tool track or store training data?
menu:
  support:
    identifier: ja-support-tool_track_store_training_data
tags:
- experiments
toc_hide: true
type: docs
---

SHA または一意の識別子を `wandb.config.update(...)` に渡して、データセットをトレーニング run に関連付けます。W&B は、ローカル・ファイル名で `wandb.save` が呼び出されない限り、データを保存しません。
