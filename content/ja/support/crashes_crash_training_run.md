---
menu:
  support:
    identifier: ja-support-crashes_crash_training_run
tags:
- crashing and hanging runs
title: If wandb crashes, will it possibly crash my training run?
toc_hide: true
type: docs
---

It is critical to avoid interference with training runs. W&B operates in a separate process, ensuring that training continues even if W&B experiences a crash. In the event of an internet outage, W&B continually retries sending data to [wandb.ai](https://wandb.ai).