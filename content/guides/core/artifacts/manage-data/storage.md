---
description: Manage storage, memory allocation of W&B Artifacts.
menu:
  default:
    identifier: storage
    parent: manage-data
title: Manage artifact storage and memory allocation
---

W&B stores artifact files in a private Google Cloud Storage bucket located in the United States by default. All files are encrypted at rest and in transit.

For sensitive files, we recommend you set up [Private Hosting]({{< relref "/guides/hosting/" >}}) or use [reference artifacts]({{< relref "../track-external-files.md" >}}).

During training, W&B locally saves logs, artifacts, and configuration files in the following local directories:

| File | Default location | To change default location set: |
| ---- | ---------------- | ------------------------------- |
| logs | `./wandb` | `dir` in `wandb.init` or set the `WANDB_DIR` environment variable |
| artifacts | `~/.cache/wandb` | the `WANDB_CACHE_DIR` environment variable |
| configs | `~/.config/wandb` | the `WANDB_CONFIG_DIR` environment variable |
| staging artifacts for upload  | `~/.cache/wandb-data/` | the `WANDB_DATA_DIR` environment variable |
| downloaded artifacts | `./artifacts` | the `WANDB_ARTIFACT_DIR` environment variable |

For a complete guide to using environment variables to configure W&B, see the [environment variables reference]({{< relref "/guides/models/track/environment-variables.md" >}}).

{{% alert color="secondary" %}}
Depending on the machine on `wandb` is initialized on, these default folders may not be located in a writeable part of the file system. This might trigger an error.
{{% /alert %}}

### Clean up local artifact cache

W&B caches artifact files to speed up downloads across versions that share files in common. Over time this cache directory can become large. Run the [`wandb artifact cache cleanup`]({{< relref "/ref/cli/wandb-artifact/wandb-artifact-cache/" >}}) command to prune the cache and to remove any files that have not been used recently.

The proceeding code snippet demonstrates how to limit the size of the cache to 1GB. Copy and paste the code snippet into your terminal:

```bash
$ wandb artifact cache cleanup 1GB
```
