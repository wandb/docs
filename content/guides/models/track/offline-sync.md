---
description: Use WANDB_MODE=offline and wandb sync to do offline training and sync to W&B later
menu:
  default:
    identifier: offline-sync
    parent: experiments
weight: 2
title: Offline training and synchronization
---

You can use offline training and synchronization for cases where you may want to perform training tasks offline, such as: 

* If you don’t have internet
* If you want to run training tasks without it being logged upstream
* If you want to avoid using resources on a particular training machine

## Workflow for offline training

1. Set the environment variable `WANDB_MODE=offline` to save the metrics locally, no internet required.
2. When you're ready, run [`wandb init`]({{< relref "/ref/cli/wandb-init/" >}}) in your directory to set the project name.
3. Run [`wandb sync YOUR_RUN_DIRECTORY`]({{< relref "/ref/cli/wandb-sync/" >}}) to push the metrics to our cloud service and see your results in our hosted web app.

You can check via API whether your run is offline by using `run.settings._offline` or `run.settings.mode` after running `wandb.init()`.

