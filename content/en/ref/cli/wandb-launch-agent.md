---
title: wandb launch-agent
---

Run a W&B launch agent.

## Usage

```bash
wandb launch-agent [OPTIONS]
```

## Options

| Option | Description |
| :--- | :--- |
| `--queue`, `-q` | The name of a queue for the agent to watch. Multiple -q flags supported. |
| `--entity`, `-e` | The entity to use. Defaults to current logged-in user |
| `--log-file`, `-l` | Destination for internal agent logs. Use - for stdout. By default all agents logs will go to debug.log in your wandb/ subdirectory or WANDB_DIR if set. |
| `--max-jobs`, `-j` | The maximum number of launch jobs this agent can run in parallel. Defaults to 1. Set to -1 for no upper limit |
| `--config`, `-c` | path to the agent config yaml to use |
| `--url`, `-u` | a wandb client registration URL, this is generated in the UI |
| `--verbose`, `-v` | Display verbose output (default: 0) |
