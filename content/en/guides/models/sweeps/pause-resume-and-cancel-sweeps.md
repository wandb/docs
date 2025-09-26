---
description: Pause, resume, and cancel a W&B Sweep with the CLI.
menu:
  default:
    identifier: pause-resume-and-cancel-sweeps
    parent: sweeps
title: Manage a W&B Sweep with the CLI
weight: 8
---

Use the [W&B CLI]({{< relref "/ref/cli/wandb-sweep.md" >}}) to pause, resume, and cancel a sweep. The CLI's `sweep` command uses flags such as `--pause` and `--resume` to control the sweep's ability to create new W&B runs, with different effects on existing runs:

- **Pause**: When you pause a sweep, the agent creates no new runs until you resume the sweep. Existing runs continue to execute normally.
- **Resume**: When you resume a sweep, the agent continues creating new runs according to the search strategy.
- **Stop**: When you stop a sweep, the agent stops creating new runs. Existing runs continue to completion.
- **Cancel**: When you cancel a sweep, the agent immediately kills all currently executing runs and stops creating new runs.

In each case, provide the sweep ID that was generated when you initialized a sweep. Optionally open a new terminal window to execute the proceeding commands. A new terminal window makes it easier to execute a command if a sweep is printing output statements to your current terminal window.

Use the following guidance to pause, resume, and cancel a sweep.

### Pause a sweep

Pause a sweep so it temporarily stops creating new runs. Runs that are already executing will continue to run until completion. Use the [`wandb sweep --pause`]({{< relref "/ref/cli/wandb-sweep.md" >}}) command to pause a sweep. Provide the sweep ID that you want to pause.

```bash
wandb sweep --pause entity/project/sweep_ID
```

### Resume a sweep

Resume a paused sweep with the [`wandb sweep --resume`]({{< relref "/ref/cli/wandb-sweep.md" >}}) command. The sweep will start creating new runs again according to its search strategy. Provide the sweep ID that you want to resume:

```bash
wandb sweep --resume entity/project/sweep_ID
```

### Stop a sweep

Finish a sweep to stop creating new runs while letting currently executing runs finish gracefully. Use the [`wandb sweep --stop`]({{< relref "/ref/cli/wandb-sweep.md" >}}) command:

```bash
wandb sweep --stop entity/project/sweep_ID
```

### Cancel a sweep

Cancel a sweep to immediately kill all running runs and stop creating new runs. This is the only sweep command that forcibly terminates existing runs. Use the [`wandb sweep --cancel`]({{< relref "/ref/cli/wandb-sweep.md" >}}) command to cancel a sweep. Provide the sweep ID that you want to cancel.

```bash
wandb sweep --cancel entity/project/sweep_ID
```

For a full list of CLI command options, see the [wandb sweep]({{< relref "/ref/cli/wandb-sweep.md" >}}) CLI Reference Guide.

## Understanding sweep and run statuses

A W&B Sweep orchestrates multiple W&B Runs to explore hyperparameter combinations. Understanding how sweep status and run status interact is crucial for effectively managing your hyperparameter optimization.

### Key differences

- **Sweep status** controls whether new runs are created (Running, Paused, Stopped, Cancelled, Finished, Failed, Crashed)
- **Run status** reflects the execution state of individual runs (Pending, Running, Finished, Failed, Crashed, Killed)
- Sweep commands affect run generation, not run execution (except cancel)

### How commands affect existing runs

| Command | Sweep Status Change | New Runs | Existing Runs |
|---------|-------------------|----------|---------------|
| `--pause` | → Paused | Stops creating | Continue normally |
| `--resume` | Paused → Running | Resumes creating | No effect |
| `--stop` | → Stopped | Stops creating | Continue to completion |
| `--cancel` | → Cancelled | Stops creating | Immediately killed |

### Common scenarios

**Individual run failures**: When runs fail within a sweep, the run status shows as `Failed` but the sweep status remains `Running`. The sweep continues creating new runs, and the search algorithm may learn from the failure.

**Resource constraints**: On preemptible resources, runs may show as `Crashed` if the instance is preempted, but the sweep status remains `Running`. Use `run.mark_preempting()` to automatically requeue crashed runs.

**Partial analysis**: You can pause a sweep to analyze completed runs while others finish, then decide whether to resume, stop, or cancel based on results.

### Best practices

- Use pause instead of cancel when you want to temporarily halt exploration without losing running experiments
- Monitor individual run statuses to identify systematic failures
- Use stop for graceful termination when you've found satisfactory hyperparameters
- Reserve cancel for emergencies when runs are consuming excessive resources or producing errors

```bash
wandb sweep --pause entity/project/sweep_ID
```

Specify the `--resume` flag along with the sweep ID to resume the sweep across your agents:

```bash
wandb sweep --resume entity/project/sweep_ID
```

For more information on how to parallelize agents, see [Parallelize agents]({{< relref "./parallelize-agents.md" >}}).