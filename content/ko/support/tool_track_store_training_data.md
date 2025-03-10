---
menu:
  support:
    identifier: ko-support-tool_track_store_training_data
tags:
- experiments
title: Does your tool track or store training data?
toc_hide: true
type: docs
---

Pass a SHA or unique identifier to `wandb.config.update(...)` to associate a dataset with a training run. W&B stores no data unless `wandb.save` is called with the local file name.