---
title: "My run's state is "crashed" on the UI but is still running on my machine. What do I do to get my data back?"
tags: []
---

### My run's state is "crashed" on the UI but is still running on my machine. What do I do to get my data back?
You most likely lost connection to your machine while training. You can recover your data by running [`wandb sync [PATH_TO_RUN]`](../../ref/cli/wandb-sync.md). The path to your run will be a folder in your `wandb` directory corresponding to the Run ID of the run in progress.