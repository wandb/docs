---
menu:
  support:
    identifier: ja-support-kb-articles-tool_track_store_training_data
support:
- experiments
title: Does your tool track or store training data?
toc_hide: true
type: docs
url: /support/:filename
---

Pass a SHA or unique identifier to `wandb.config.update(...)` to associate a dataset with a training run. W&B stores no data unless `wandb.save` is called with the local file name.