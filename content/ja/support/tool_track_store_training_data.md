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

`wandb.config.update(...)` に SHA またはユニークな識別子を渡して、データセット を training run に関連付けます。W&B は、ローカルファイル名と共に `wandb.save` が呼ばれない限り、データ は保存されません。