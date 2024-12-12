---
title: "How can I use wandb with multiprocessing, e.g. distributed training?"
toc_hide: true
type: docs
tags:
   - experiments
---
If a training program uses multiple processes, structure the program to avoid making wandb method calls from processes without `wandb.init()`. 

Manage multiprocess training using these approaches:

1. Call `wandb.init` in all processes and use the [group](../guides/runs/grouping.md) keyword argument to create a shared group. Each process will have its own wandb run, and the UI will group the training processes together.
2. Call `wandb.init` from only one process and pass data to log through [multiprocessing queues](https://docs.python.org/3/library/multiprocessing.html#exchanging-objects-between-processes).

{{% alert %}}
Refer to the [Distributed Training Guide](../guides/track/log/distributed-training.md) for detailed explanations of these approaches, including code examples with Torch DDP.
{{% /alert %}}