---
title: wandb launch-sweep
---

Run a W&B launch sweep (Experimental).

## Usage

```bash
wandb launch-sweep [CONFIG] [OPTIONS]
```

## Arguments

| Argument | Description | Required |
| :--- | :--- | :--- |
| `CONFIG` | No description available | No |

## Options

| Option | Description |
| :--- | :--- |
| `--queue`, `-q` | The name of a queue to push the sweep to |
| `--project`, `-p` | Name of the project which the agent will watch. If passed in, will override the project value passed in using a config file |
| `--entity`, `-e` | The entity to use. Defaults to current logged-in user |
| `--resume_id`, `-r` | Resume a launch sweep by passing an 8-char sweep id. Queue required |
| `--prior_run`, `-R` | ID of an existing run to add to this sweep |
