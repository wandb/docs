---
url: /support/:filename
title: "Does your tool track or store training data?"
toc_hide: true
type: docs
support:
   - experiments
---
Pass a SHA or unique identifier to `wandb.Run.config.update(...)` to associate a dataset with a training run. W&B stores no data unless `wandb.Run.save()` is called with the local file name.