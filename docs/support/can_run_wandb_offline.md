---
title: "Can I run wandb offline?"
tags:
   - None
---
If training occurs on an offline machine, use the following steps to upload results to the servers:

1. Set the environment variable `WANDB_MODE=offline` to save metrics locally without an internet connection.
2. When ready to upload, run `wandb init` in your directory to set the project name. 
3. Use `wandb sync YOUR_RUN_DIRECTORY` to transfer metrics to the cloud service and access results in the hosted web app.

To confirm the run is offline, check `run.settings._offline` or `run.settings.mode` after executing `wandb.init()`.