---
title: "Does your tool track or store training data?"
tags:
   - None
---
Pass a SHA or unique identifier to `wandb.config.update(...)` to associate a dataset with a training run. W&B stores no data unless `wandb.save` is called with the local file name.