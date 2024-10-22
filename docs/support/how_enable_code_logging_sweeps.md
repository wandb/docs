---
title: "How do I enable code logging with Sweeps?"
tags:
   - sweeps
---

To enable code logging for sweeps, simply add `wandb.log_code()` after you have initialized your W&B Run. This is necessary even when you have enabled code logging in the settings page of your W&B profile in the app. For more advanced code logging, see the [docs for `wandb.log_code()` here](../ref/python/run.md#log_code).