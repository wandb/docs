---
title: wandb beta sync
---

**Usage**

`wandb beta sync [OPTIONS] [PATHS]...`

**Summary**

Upload .wandb files specified by PATHS.

PATHS can include .wandb files, run directories containing .wandb files, and
"wandb" directories containing run directories.

For example, to sync all runs in a directory:
wandb beta sync ./wandb
To sync a specific run:
wandb beta sync ./wandb/run-20250813_124246-n67z9ude
Or equivalently:
wandb beta sync ./wandb/run-20250813_124246-n67z9ude/run-n67z9ude.wandb

**Options**

| **Option** | **Description** |
| :--- | :--- |
| `--skip-synced / --no-skip-synced` | Skip runs that have already been synced with this command. |
| `--dry-run` | Print what would happen without uploading   anything. |
| `-v, --verbose` | Print more information. |
| `-n` | Max number of runs to sync at a time. |



