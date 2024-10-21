---
title: "How does wandb stream logs and writes to disk?"
tags: []
---

### How does wandb stream logs and writes to disk?
W&B queues in memory but also [write the events to disk](https://github.com/wandb/wandb/blob/7cc4dd311f3cdba8a740be0dc8903075250a914e/wandb/sdk/internal/datastore.py) asynchronously to handle failures and for the `WANDB_MODE=offline` case where you can sync the data after it's been logged.

In your terminal, you can see a path to the local run directory. This directory will contain a `.wandb` file that is the datastore above. If you're also logging images, we write them to `media/images` in that directory before uploading them to cloud storage.