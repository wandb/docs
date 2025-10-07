---
title: wandb sync
---

**Usage**

`wandb sync [OPTIONS] [PATH]...`

**Summary**

Synchronize W&B run data to the cloud.

If PATH is provided, sync runs found at the given path. If a path is not
specified, search for `./wandb` first, then search for a `wandb/`
subdirectory.

To sync a specific run:
wandb sync ./wandb/run-20250813_124246-n67z9ude
Or equivalently:
wandb sync ./wandb/run-20250813_124246-n67z9ude/run-n67z9ude.wandb

**Options**

| **Option** | **Description** |
| :--- | :--- |
| `--id` | The run you want to upload to. |
| `-p, --project` | The project you want to upload to. |
| `-e, --entity` | The entity to scope to. |
| `--job_type` | Specifies the type of run for grouping   related runs together. |
| `--sync-tensorboard / --no-sync-tensorboard` | Stream tfevent files to wandb. |
| `--include-globs` | Comma separated list of globs to include. |
| `--exclude-globs` | Comma separated list of globs to exclude. |
| `--include-online / --no-include-online` | Include online runs |
| `--include-offline / --no-include-offline` | Include offline runs |
| `--include-synced / --no-include-synced` | Include synced runs |
| `--mark-synced / --no-mark-synced` | Mark runs as synced |
| `--sync-all` | Sync all runs |
| `--clean` | Delete synced runs |
| `--clean-old-hours` | Delete runs created before this many hours.   To be used alongside --clean flag. |
| `--clean-force` | Clean without confirmation prompt. |
| `--show` | Number of runs to show |
| `--append` | Append run |
| `--skip-console` | Skip console logs |
| `--replace-tags` | Replace tags in the format   'old_tag1=new_tag1,old_tag2=new_tag2' |



