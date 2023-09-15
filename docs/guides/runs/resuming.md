---
description: Resume a paused or exited W&B Run
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Resume Runs

<head>
  <title>Resume W&B Runs</title>
</head>

This page covers how to enable W&B to automatically resume runs that have crashed or finished.

<!-- If the `resume` parameter is left unspecified W&B will, by default, create a new run and overwrite the data of the crashed run if you start a new run that has the same run ID as the run that crashed. -->

## Automatically resume runs
Enable W&B to automatically resume runs.

### Resume a run without overriding the stopped or crashed run
Set the `resume` parameter to True (`resume=True`) when you initialize a run with W&B (`wandb.init`). 

```python
run = wandb.init(resume=True)
```

:::note
This only works if you run your script in the same directory as the one that failed as the file is stored at: `wandb/wandb-resume.json`.
:::

### Resume an identical run with the same run ID
Determine how W&B resumes identical runs based on the run ID. Define the run ID either with the `id` parameter when you initialize a run with `wandb.intit` or set the`WANDB_RUN_ID` environment variable. Ensure that you provide the ID of the run, the project the ID belongs to, along with the project the run belongs to.

Based on your use case, specify one of the following:
- `"allow"`:  W&B automatically resumes the run with the run ID specified. Otherwise, W&B will start a new run.
- `"never"`:  W&B will crash.
- `"must"`:   W&B automatically resumes the run with the ID specified. Otherwise, W&B will crash.

For example:

```python
run = wandb.init(entity="<entity>", project="<project>", id="<run ID>", resume="must")
```


<!-- START -->

### Automatic and controlled resuming

Automatic resuming only works if the process is restarted on top of the same filesystem as the failed process. Set the `WANDB_RUN_ID` environment variable if you can not share a filesystem. For example:


```python
# store this id to use it later when resuming
id = wandb.util.generate_id()
wandb.init(id=id, resume="allow")
# or via environment variables
os.environ["WANDB_RESUME"] = "allow"
os.environ["WANDB_RUN_ID"] = wandb.util.generate_id()
wandb.init()
```

If you set `WANDB_RESUME` equal to `"allow"`, you can always set `WANDB_RUN_ID` to a unique string and restarts of the process will be handled automatically. If you set `WANDB_RESUME` equal to `"must"`, W&B will throw an error if the run to be resumed does not exist yet instead of auto-creating a new run.

:::caution
If multiple processes use the same `run_id` concurrently unexpected results will be recorded and rate limiting will occur.
:::

:::info
If you resume a run and you have `notes` specified in `wandb.init()`, those notes will overwrite any notes that you have added in the UI.
:::

<!-- END -->




## Resume preemptible Sweeps runs
Automatically requeue interrupted sweep runs. This is particularly useful if you run a sweep agent in a compute environment that is subject to preemption (for example, a SLURM job in a preemptible queue, an EC2 spot instance, or a Google Cloud preemptible VM).

Use the [`mark_preempting`](../../ref/python/run.md#markpreempting) to enable W&B to automatically requeue interrupted sweep runs:

```python
wandb.mark_preempting()
```

If a run that is marked preempting exits with status code 0, W&B will consider the run to have terminated successfully and it will not be requeued. If a preempting run exits with a nonzero status, W&B will consider the run to have been preempted, and it will automatically append the run to a run queue associated with the sweep. If a run exits with no status, W&B will mark the run preempted 5 minutes after the run's final heartbeat, then add it to the sweep run queue. Sweep agents will consume runs off the run queue until the queue is exhausted, at which point they will resume generating new runs based on the standard sweep search algorithm.

:::tip
By default, requeued runs begin logging from their initial step. To instruct a run to resume logging at the step where it was interrupted, initialize the resumed run with `wandb.init(resume=True)`. 
:::

