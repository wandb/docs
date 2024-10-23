---
title: "How does wandb stream logs and writes to disk?"
displayed_sidebar: support
tags:
   - environment variables
---
W&B queues events in memory and writes them to disk asynchronously to manage failures and support the `WANDB_MODE=offline` configuration, allowing synchronization after logging.

In the terminal, observe the path to the local run directory. This directory includes a `.wandb` file, which serves as the datastore. For image logging, W&B stores images in the `media/images` subdirectory before uploading them to cloud storage.