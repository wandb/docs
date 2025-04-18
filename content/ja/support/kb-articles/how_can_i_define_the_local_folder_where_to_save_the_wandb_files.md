---
menu:
  support:
    identifier: ja-support-kb-articles-how_can_i_define_the_local_folder_where_to_save_the_wandb_files
support:
- environment variables
- experiments
title: How can I define the local location for `wandb` files?
toc_hide: true
type: docs
url: /support/:filename
---

- `WANDB_DIR=<path>` or `wandb.init(dir=<path>)`: Controls the location of the `wandb` folder created for your training script. Defaults to `./wandb`. This folder stores Run's data and logs
- `WANDB_ARTIFACT_DIR=<path>` or `wandb.Artifact().download(root="<path>")`: Controls the location where artifacts are downloaded. Defaults to `./artifacts`
- `WANDB_CACHE_DIR=<path>`: This is the location where artifacts are created and stored when you call `wandb.Artifact`. Defaults to `~/.cache/wandb`
- `WANDB_CONFIG_DIR=<path>`: Where config files are stored. Defaults to `~/.config/wandb`
- `WANDB_DATA_DIR=<PATH>`: Controls the location used for staging artifacts during upload. Defaults to `~/.cache/wandb-data/`.