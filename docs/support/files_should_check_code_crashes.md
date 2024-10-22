---
title: "Which files should I check when my code crashes?"
tags:
   - None
---
For the affected run, check `debug.log` and `debug-internal.log`. These files are located in the local folder `wandb/run-<date>_<time>-<run-id>/logs`, which is in the same directory as the running code.