---
title: wandb sync
---

Upload an offline training directory to W&B

## Usage

```bash
wandb sync [PATH] [OPTIONS]
```

## Arguments

| Argument | Description | Required |
| :--- | :--- | :--- |
| `PATH` | No description available | No |

## Options

| Option | Description |
| :--- | :--- |
| `--view` | View runs (default: False) |
| `--verbose` | Verbose (default: False) |
| `--id` | The run you want to upload to. |
| `--project`, `-p` | The project you want to upload to. |
| `--entity`, `-e` | The entity to scope to. |
| `--job_type` | Specifies the type of run for grouping related runs together. |
| `--sync-tensorboard` | Stream tfevent files to wandb. |
| `--include-globs` | Comma separated list of globs to include. |
| `--exclude-globs` | Comma separated list of globs to exclude. |
| `--include-online` | Include online runs |
| `--include-offline` | Include offline runs |
| `--include-synced` | Include synced runs |
| `--mark-synced` | Mark runs as synced (default: True) |
| `--sync-all` | Sync all runs (default: False) |
| `--clean` | Delete synced runs (default: False) |
| `--clean-old-hours` | Delete runs created before this many hours. To be used alongside --clean flag. (default: 24) |
| `--clean-force` | Clean without confirmation prompt. (default: False) |
| `--ignore` | No description available |
| `--show` | Number of runs to show (default: 5) |
| `--append` | Append run (default: False) |
| `--skip-console` | Skip console logs (default: False) |
| `--replace-tags` | Replace tags in the format 'old_tag1=new_tag1,old_tag2=new_tag2' |
