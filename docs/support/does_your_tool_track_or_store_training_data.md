---
title: "Does your tool track or store training data?"
tags: []
---

### Does your tool track or store training data?
You can pass a SHA or other unique identifier to `wandb.config.update(...)` to associate a dataset with a training run. W&B does not store any data unless `wandb.save` is called with the local file name.