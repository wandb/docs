---
title: "What does wandb.init do to my training process?"
displayed_sidebar: support
tags:
   - environment variables
   - experiments
---
When `wandb.init()` runs in a training script, an API call creates a run object on the servers. A new process starts to stream and collect metrics, allowing the primary process to function normally. The script writes to local files while the separate process streams data to the servers, including system metrics. To turn off streaming, run `wandb off` from the training directory or set the `WANDB_MODE` environment variable to `offline`.