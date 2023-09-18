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

This page covers how to automatically resume runs that crashed or finished. If you do not specify that you want runs to automatically resume crashed or finished runs, W&B will (by default), create a new run. This new run will overwrite the data of the crashed run if you the new run has the same run ID.

:::caution
Unexpected results will occur if multiple processes use the same `run_id` concurrently. 
:::

:::info
* If you resume a run and specify `notes` in `wandb.init()`, those notes will overwrite any notes that you make in the W&B App UI.
:::


## Enable runs to automatically resume 
Automatic resuming only works if the process is restarted on top of the same filesystem as the failed process. 

<!-- This only works if you run your script in the same directory as the one that failed as the file is stored at: `wandb/wandb-resume.json`. -->


If you can not share a filesystem, specify the `WANDB_RUN_ID` environment variable or pass the run ID with the W&B Python SDK. See the [Create a run](./intro.md#create-a-run) section in the "What are runs?" page for more information on run IDs.


<Tabs
  defaultValue="python"
  values={[
    {label: 'W&B Python SDK', value: 'python'},
    {label: 'Shell script', value: 'bash'},
  ]}>
  <TabItem value="python">

The following code snippet shows how to specify a W&B run ID with the Python SDK. Your code will look similar to the following code snippet. Replace values enclosed within `<>` with your own:

```python
run = wandb.init(
  entity="<entity>", project="<project>", 
  id="<run ID>", resume="must"
  )
```

  </TabItem>
  <TabItem value="bash">

The following example shows how to specify the W&B `WANDB_RUN_ID` variable in a bash script: 

```bash title="run_experiment.sh"
RUN_ID="$1"

WANDB_RESUME=allow WANDB_RUN_ID="$RUN_ID" python eval.py
```
Within your terminal, you could run the shell script along with the W&B run ID. The following code snippet passes the run ID `akj172`: 

```bash
sh run_experiment.sh akj172 
```

  </TabItem>
</Tabs>



## Resume a run without overriding the existing run
Resume a run that stopped or crashed without overriding the existing run. This is especially helpful if  if your process doesn't exit successfully. The next time you start W&B, W&B will start logging from the last step.

Set the `resume` parameter to True (`resume=True`) when you initialize a run with W&B (`wandb.init`). The following code snippet shows how to accomplish this with the W&B Python SDK:

```python
import wandb

run = wandb.init(
  entity="<entity>", 
  project="<project>", 
  resume=True
  )
```

## Resume a run using the same run ID
Resume a run that uses the same run ID when it crashed or failed. To do so, ensure you satisfy the following requirements when you initialize a run with `wandb.init`:

1. Define the run ID. Pass a unique identifier to the `id` parameter (`id`).
2. Specify a project for the run (`project` parameter).
3. Specify one of three options for the `resume` parameter; `allow`, `never`, or `must`. They are defined as follows:
  - `"allow"`:  W&B checks if the run exists. If the run exists, W&B uses that run. Otherwise, W&B initializes a new run with the specified run ID. 
  - `"never"`: W&B checks if the run exists. If the run exists, the run will fail. Otherwise, W&B will start a new run with the specified run id.
  - `"must"`: W&B checks if the run exists. If the run exists, W&B uses that run. Otherwise, the run will fail.


The following code snippet shows how to resume  a run that uses the same run ID with the W&B Python SDK. Your code will look similar to the following code snippet. 

Replace values enclosed within `<>` with your own:

```python
run = wandb.init(
  entity="<entity>", project="<project>", 
  id="<run ID>", resume="must"
  )
```




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



