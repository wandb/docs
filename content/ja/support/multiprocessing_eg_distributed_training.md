---
menu:
  support:
    identifier: ja-support-multiprocessing_eg_distributed_training
tags:
- experiments
title: How can I use wandb with multiprocessing, e.g. distributed training?
toc_hide: true
type: docs
---

If a training program uses multiple processes, structure the program to avoid making wandb method calls from processes without `wandb.init()`. 

Manage multiprocess training using these approaches:

1. Call `wandb.init` in all processes and use the [group]({{< relref path="/guides/models/track/runs/grouping.md" lang="ja" >}}) keyword argument to create a shared group. Each process will have its own wandb run, and the UI will group the training processes together.
2. Call `wandb.init` from only one process and pass data to log through [multiprocessing queues](https://docs.python.org/3/library/multiprocessing.html#exchanging-objects-between-processes).

{{% alert %}}
Refer to the [Distributed Training Guide]({{< relref path="/guides/models/track/log/distributed-training.md" lang="ja" >}}) for detailed explanations of these approaches, including code examples with Torch DDP.
{{% /alert %}}