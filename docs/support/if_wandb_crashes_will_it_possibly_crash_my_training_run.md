---
title: "If wandb crashes, will it possibly crash my training run?"
tags: []
---

### If wandb crashes, will it possibly crash my training run?
It is extremely important to us that we never interfere with your training runs. We run wandb in a separate process to make sure that if wandb somehow crashes, your training will continue to run. If the internet goes out, wandb will continue to retry sending data to [wandb.ai](https://wandb.ai).