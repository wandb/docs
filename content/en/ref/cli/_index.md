---
title: Command Line Interface
weight: 2
no_list: true
---

## Usage

```bash
wandb [OPTIONS] COMMAND [ARGS]...
```

## Options

| Option | Description |
| :--- | :--- |
| `--version` | Show the version and exit. (default: False) |

## Commands

| Command | Description |
| :--- | :--- |
| [agent]({{< relref "wandb-agent" >}}) | Run the W&B agent |
| [artifact]({{< relref "wandb-artifact" >}}) | Commands for interacting with artifacts |
| [beta]({{< relref "wandb-beta" >}}) | Beta versions of wandb CLI commands. |
| [controller]({{< relref "wandb-controller" >}}) | Run the W&B local sweep controller |
| [disabled]({{< relref "wandb-disabled" >}}) | Disable W&B. |
| [docker]({{< relref "wandb-docker" >}}) | Run your code in a docker container. |
| [docker-run]({{< relref "wandb-docker-run" >}}) | Wrap `docker run` and adds WANDB_API_KEY and WANDB_DOCKER environment variables. |
| [enabled]({{< relref "wandb-enabled" >}}) | Enable W&B. |
| [init]({{< relref "wandb-init" >}}) | Configure a directory with Weights & Biases |
| [job]({{< relref "wandb-job" >}}) | Commands for managing and viewing W&B jobs |
| [launch]({{< relref "wandb-launch" >}}) | Launch or queue a W&B Job. |
| [launch-agent]({{< relref "wandb-launch-agent" >}}) | Run a W&B launch agent. |
| [launch-sweep]({{< relref "wandb-launch-sweep" >}}) | Run a W&B launch sweep (Experimental). |
| [local]({{< relref "wandb-local" >}}) | Start a local W&B container (deprecated, see wandb server --help) |
| [login]({{< relref "wandb-login" >}}) | Login to Weights & Biases |
| [off]({{< relref "wandb-off" >}}) | No description available |
| [offline]({{< relref "wandb-offline" >}}) | Disable W&B sync |
| [on]({{< relref "wandb-on" >}}) | No description available |
| [online]({{< relref "wandb-online" >}}) | Enable W&B sync |
| [projects]({{< relref "wandb-projects" >}}) | List projects |
| [pull]({{< relref "wandb-pull" >}}) | Pull files from Weights & Biases |
| [restore]({{< relref "wandb-restore" >}}) | Restore code, config and docker state for a run. |
| [scheduler]({{< relref "wandb-scheduler" >}}) | Run a W&B launch sweep scheduler (Experimental) |
| [server]({{< relref "wandb-server" >}}) | Commands for operating a local W&B server |
| [status]({{< relref "wandb-status" >}}) | Show configuration settings |
| [sweep]({{< relref "wandb-sweep" >}}) | Initialize a hyperparameter sweep. |
| [sync]({{< relref "wandb-sync" >}}) | Upload an offline training directory to W&B |
| [verify]({{< relref "wandb-verify" >}}) | Checks and verifies local instance of W&B. |
