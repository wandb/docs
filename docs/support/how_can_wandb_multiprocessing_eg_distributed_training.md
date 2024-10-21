---
title: "How can I use wandb with multiprocessing, e.g. distributed training?"
tags:
   - experiments
---

If your training program uses multiple processes you will need to structure your program to avoid making wandb method calls from processes where you did not run `wandb.init()`.\
\
There are several approaches to managing multiprocess training:

1. Call `wandb.init` in all your processes, using the [group](../runs/grouping.md) keyword argument to define a shared group. Each process will have its own wandb run and the UI will group the training processes together.
2. Call `wandb.init` from just one process and pass data to be logged over [multiprocessing queues](https://docs.python.org/3/library/multiprocessing.html#exchanging-objects-between-processes).

:::info
Check out the [Distributed Training Guide](./log/distributed-training.md) for more detail on these two approaches, including code examples with Torch DDP.
:::