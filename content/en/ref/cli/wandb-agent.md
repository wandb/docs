---
title: wandb agent
---

Run the W&B agent

## Usage

```bash
wandb agent SWEEP_ID [OPTIONS]
```

## Arguments

| Argument | Description | Required |
| :--- | :--- | :--- |
| `SWEEP_ID` | No description available | Yes |

## Options

| Option | Description |
| :--- | :--- |
| `--project`, `-p` | The name of the project where W&B runs created from the sweep are sent to. If the project is not specified, the run is sent to a project labeled 'Uncategorized'. |
| `--entity`, `-e` | The username or team name where you want to send W&B runs created by the sweep to. Ensure that the entity you specify already exists. If you don't specify an entity, the run will be sent to your default entity, which is usually your username. |
| `--count` | The max number of runs for this agent. |
