---
title: "How do I enable code logging with Sweeps?"
displayed_sidebar: support
tags:
   - sweeps
---
To enable code logging for sweeps, add `wandb.log_code()` after initializing the W&B Run. This action is necessary even when code logging is enabled in the W&B profile settings. For advanced code logging, refer to the [docs for `wandb.log_code()` here](../ref/python/run.md#log_code).