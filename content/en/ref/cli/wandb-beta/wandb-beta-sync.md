---
title: wandb beta sync
---

Upload a training run to W&B

## Usage

```bash
wandb sync WANDB_DIR [OPTIONS]
```

## Arguments

| Argument | Description | Required |
| :--- | :--- | :--- |
| `WANDB_DIR` | No description available | Yes |

## Options

| Option | Description |
| :--- | :--- |
| `--id` | The run you want to upload to. |
| `--project`, `-p` | The project you want to upload to. |
| `--entity`, `-e` | The entity to scope to. |
| `--skip-console` | Skip console logs (default: False) |
| `--append` | Append run (default: False) |
| `--include`, `-i` | Glob to include. Can be used multiple times. |
| `--exclude`, `-e` | Glob to exclude. Can be used multiple times. |
| `--mark-synced` | Mark runs as synced (default: True) |
| `--skip-synced` | Skip synced runs (default: True) |
| `--dry-run` | Perform a dry run without uploading anything. (default: False) |
