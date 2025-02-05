---
title: How can I define the local folder where to save the wandb files?
toc_hide: true
type: docs
tags:
  - environment variables
  - experiments
---

- `WANDB_DIR=<path>` or `wandb.init(dir=<path>)`: Controls the location of the `wandb` folder created for your training script. Defaults to `./wandb`
- `WANDB_ARTIFACT_DIR=<path>` or `wandb.Artifact().download(root="<path>")`: Controls the location where artifacts are downloaded. Defaults to `artifacts/`
- `WANDB_CACHE_DIR=<path>`: This is the location where artifacts are created and stored when you call `wandb.Artifact`. Defaults to `~/.cache/wandb`
- `WANDB_CONFIG_DIR=<path>`: Where config files are stored. Defaults to `~/.config/wandb`