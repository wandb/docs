---
title: wandb sweep
---

Initialize a hyperparameter sweep. Search for hyperparameters that optimizes a cost function of a machine learning model by testing various combinations.

## Usage

```bash
wandb sweep CONFIG_YAML_OR_SWEEP_ID [OPTIONS]
```

## Arguments

| Argument | Description | Required |
| :--- | :--- | :--- |
| `CONFIG_YAML_OR_SWEEP_ID` | No description available | Yes |

## Options

| Option | Description |
| :--- | :--- |
| `--project`, `-p` | The name of the project where W&B runs created from the sweep are sent to. If the project is not specified, the run is sent to a project labeled Uncategorized. |
| `--entity`, `-e` | The username or team name where you want to send W&B runs created by the sweep to. Ensure that the entity you specify already exists. If you don't specify an entity, the run will be sent to your default entity, which is usually your username. |
| `--controller` | Run local controller (default: False) |
| `--verbose` | Display verbose output (default: False) |
| `--name` | The name of the sweep. The sweep ID is used if no name is specified. |
| `--program` | Set sweep program |
| `--settings` | Set sweep settings |
| `--update` | Update pending sweep |
| `--stop` | Finish a sweep to stop running new runs and let currently running runs finish. (default: False) |
| `--cancel` | Cancel a sweep to kill all running runs and stop running new runs. (default: False) |
| `--pause` | Pause a sweep to temporarily stop running new runs. (default: False) |
| `--resume` | Resume a sweep to continue running new runs. (default: False) |
| `--prior_run`, `-R` | ID of an existing run to add to this sweep |
