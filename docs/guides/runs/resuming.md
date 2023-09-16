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

This page covers how to automatically resume runs that have crashed or finished.

<!-- If the `resume` parameter is left unspecified W&B will, by default, create a new run and overwrite the data of the crashed run if you start a new run that has the same run ID as the run that crashed. -->

:::tip
Before you get started, ensure you have your run ID noted. 
:::

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

### Resume a run using the same run ID
Resume a run that uses the same run ID when it crashed or failed. To do so, ensure you satisfy the following requirements when you initialize a run with `wandb.init`:

1. Define the run ID. Pass a unique identifier to the `id` parameter (`id`).
2. Specify a project for the run (`project` parameter).
3. Specify one of three options for the `resume` parameter; `allow`, `never`, or `must`. They are defined as follows:
  - `"allow"`:  W&B checks if the run exists. If the run exists, W&B uses that run. Otherwise, W&B initializes a new run with the specified run ID. 
  - `"never"`: W&B checks if the run exists. If the run exists, the run will fail. Otherwise, W&B will start a new run with the specified run id.
  - `"must"`: W&B checks if the run exists. If the run exists, W&B uses that run. Otherwise, the run will fail.

Your code will look similar to the following code snippet. Replace values enclosed within `<>` with your own:

```python
run = wandb.init(
  entity="<entity>", project="<project>", 
  id="<run ID>", resume="must"
  )
```

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

:::caution
If multiple processes use the same `run_id` concurrently unexpected results will be recorded and rate limiting will occur.
:::

:::info
If you resume a run and you have `notes` specified in `wandb.init()`, those notes will overwrite any notes that you have added in the UI.
:::




## Resume preemptible Sweeps runs
Automatically requeue interrupted [sweep](../sweeps/intro.md) runs. This is particularly useful if you run a sweep agent in a compute environment that is subject to preemption such as a SLURM job in a preemptible queue, an EC2 spot instance, or a Google Cloud preemptible VM.

Use the [`mark_preempting`](../../ref/python/run.md#markpreempting) function to enable W&B to automatically requeue interrupted sweep runs. For example, the following code snippet

```python
run = wandb.init() # Initialize a run
run.mark_preempting()
```
The following table outlines how W&B handles runs based on the exit status of the a sweep run.

|Status| Behavior |
|------| ---------|
|Status code 0| Run is considered to have terminated successfully and it will not be requeued.  |
|Nonzero status| W&B automatically appends the run to a run queue associated with the sweep.|
|No status| Run is added to the sweep run queue. Sweep agents consume runs off the run queue until the queue is empty. Once the queue is empty, the sweep queue resumes generating new runs based on the sweep search algorithm.|



