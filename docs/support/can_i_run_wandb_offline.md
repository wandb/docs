---
title: "Can I run wandb offline?"
tags:
   - 
---

If you're training on an offline machine and want to upload your results to our servers afterwards, we have a feature for you!

1. Set the environment variable `WANDB_MODE=offline` to save the metrics locally, no internet required.
2. When you're ready, run `wandb init` in your directory to set the project name.
3. Run `wandb sync YOUR_RUN_DIRECTORY` to push the metrics to our cloud service and see your results in our hosted web app.

You can check via API whether your run is offline by using `run.settings._offline` or `run.settings.mode` after your wandb.init().

#