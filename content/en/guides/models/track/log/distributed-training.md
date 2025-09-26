---
description: Use W&B to log distributed training experiments with multiple GPUs.
menu:
  default:
    identifier: distributed-training
    parent: log-objects-and-media
title: Log distributed training experiments
---

During a distributed training experiment, you train a model using multiple machines or clients in parallel. W&B can help you track distributed training experiments. Based on your use case, track distributed training experiments using one of the following approaches:

* **Track a single process**: Track a rank 0 process (also known as a "leader" or "coordinator") with W&B. This is a common solution for logging distributed training experiments with the [PyTorch Distributed Data Parallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) (DDP) Class. 
* **Track multiple processes**: For multiple processes, you can either:
   * Track each process separately using one run per process. You can optionally group them together in the W&B App UI.
   * Track all processes to a single run.

<!-- The proceeding examples demonstrate how to track metrics with W&B using PyTorch DDP on two GPUs on a single machine. [PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) (`DistributedDataParallel` in`torch.nn`) is a popular library for distributed training. The basic principles apply to any distributed training setup, but the details of implementation may differ.

{{% alert %}}
Explore the code behind these examples in the W&B GitHub examples repository [here](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-ddp). Specifically, see the [`log-dpp.py`](https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py) Python script for information on how to implement one process and many process methods.
{{% /alert %}} -->

## Track a single process

This section describes how to track values and metrics available to your rank 0 process. Use this approach to track only metrics that are available from a single process. Typical metrics include GPU/CPU utilization, behavior on a shared validation set, gradients and parameters, and loss values on representative data examples.

Within the rank 0 process, initialize a W&B run with [`wandb.init()`]({{< relref "/ref/python/sdk/functions/init" >}}) and log experiments ([`wandb.log`]({{< relref "/ref/python/sdk/classes/run/#method-runlog" >}})) to that run.

The following [sample Python script (`log-ddp.py`)](https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py) demonstrates one way to track metrics on two GPUs on a single machine using PyTorch DDP. [PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) (`DistributedDataParallel` in`torch.nn`) is a popular library for distributed training. The basic principles apply to any distributed training setup, but the implementation may differ.

The Python script:
1. Starts multiple processes with `torch.distributed.launch`.
1. Checks the rank with the `--local_rank` command line argument.
1. If the rank is set to 0, sets up `wandb` logging conditionally in the [`train()`](https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py#L24) function.

```python
if __name__ == "__main__":
    # Get args
    args = parse_args()

    if args.local_rank == 0:  # only on main process
        # Initialize wandb run
        run = wandb.init(
            entity=args.entity,
            project=args.project,
        )
        # Train model with DDP
        train(args, run)
    else:
        train(args)
```

Explore an [example dashboard showing metrics tracked from a single process](https://wandb.ai/ayush-thakur/DDP/runs/1s56u3hc/system).

The dashboard displays system metrics for both GPUs, such as temperature and utilization.

{{< img src="/images/track/distributed_training_method1.png" alt="GPU metrics dashboard" >}}

However, the loss values as a function epoch and batch size were only logged from a single GPU.

{{< img src="/images/experiments/loss_function_single_gpu.png" alt="Loss function plots" >}}

## Track multiple processes

Track multiple processes with W&B with one of the following approaches:
* [Tracking each process separately]({{< relref "distributed-training/#track-each-process-separately" >}}) by creating a run for each process.
* [Tracking all processes to a single run]({{< relref "distributed-training/#track-all-processes-to-a-single-run" >}}).

### Track each process separately

This section describes how to track each process separately by creating a run for each process. Within each run you log metrics, artifacts, and forth to their respective run. Call `wandb.Run.finish()` at the end of training, to mark that the run has completed so that all processes exit properly.

You might find it difficult to keep track of runs across multiple experiments. To mitigate this, provide a value to the `group` parameter when you initialize W&B (`wandb.init(group='group-name')`) to keep track of which run belongs to a given experiment. For more information about how to keep track of training and evaluation W&B Runs in experiments, see [Group Runs]({{< relref "/guides/models/track/runs/grouping.md" >}}).

{{% alert %}}
**Use this approach if you want to track metrics from individual processes**. Typical examples include the data and predictions on each node (for debugging data distribution) and metrics on individual batches outside of the main node. This approach is not necessary to get system metrics from all nodes nor to get summary statistics available on the main node.
{{% /alert %}}

The following Python code snippet demonstrates how to set the group parameter when you initialize W&B:

```python
if __name__ == "__main__":
    # Get args
    args = parse_args()
    # Initialize run
    run = wandb.init(
        entity=args.entity,
        project=args.project,
        group="DDP",  # all runs for the experiment in one group
    )
    # Train model with DDP
    train(args, run)

    run.finish()  # mark the run as finished
```

Explore the W&B App UI to view an [example dashboard](https://wandb.ai/ayush-thakur/DDP?workspace=user-noahluna) of metrics tracked from multiple processes. Note that there are two W&B Runs grouped together in the left sidebar. Click on a group to view the dedicated group page for the experiment. The dedicated group page displays metrics from each process separately.

{{< img src="/images/experiments/dashboard_grouped_runs.png" alt="Grouped distributed runs" >}}

The preceding image demonstrates the W&B App UI dashboard. On the sidebar we see two experiments. One labeled 'null' and a second (bound by a yellow box) called 'DPP'. If you expand the group (select the Group dropdown) you will see the W&B Runs that are associated to that experiment.

### Organize distributed runs with `job_type`

Specify the `job_type` in your [`wandb_init()`]({{< relref "/ref/python/sdk/functions/init/" >}}) invocation to distinguish between different types of nodes. Then you can create saved views to help you organize your runs and filter out noise from worker nodes.

To organize your distributed training runs:

1. Set `job_type` for different node types to categorize your nodes based on their function:

   ```python
   # Main coordinating node
   with wandb.init(project="distributed-training", group="experiment_1",job_type="main") as run:
        # Training code

   # Reporting worker nodes
   with wandb.init(project="distributed-training", group="experiment_1", job_type="worker") as run:
        # Training code
   ```
1. Create [saved views]({{< relref "/guides/models/track/workspaces/#create-a-new-saved-workspace-view" >}}) in your workspace to organize your runs. First, filter your runs to show runs by their job type.  Click the **...** action menu at the top right and click **Save as new view**. For example, you could create the following saved views:

   - **Default view**: Filter out worker nodes to reduce noise
     - Click **Filter**, then set **Job Type** to `worker`.
     - Shows only your reporting nodes

   - **Debug view**: Focus on worker nodes for troubleshooting
     - Click **Filter**, then set **Job Type** `==` `worker` and set **State** to  `IN` `crashed`.
     - Shows only worker nodes that have crashed or are in error states

   - **All nodes view**: See everything together
     - No filter
     - Useful for comprehensive monitoring

To open a saved view, click **Workspaces** in the sidebar, then click the menu. Workspaces appear at the top of the list and saved views appear at the bottom.
1. Organize your workspaces for different use cases:

   - **Main workspace**: Shows only `job_type="main"` runs for high-level experiment tracking.
   - **Debug workspace**: Shows only `job_type="worker"` runs with `state != "finished"` for troubleshooting.
   - **Overview workspace**: Shows all runs grouped by `group` to see the complete experiment.

This approach gives you clean workspaces that emulate single-run behavior while maintaining visibility into all your distributed processes when needed.

### Track all processes to a single run

{{% alert color="secondary"  %}}
Parameters prefixed by `x_` (such as `x_label`) are in public preview. Create a [GitHub issue in the W&B repository](https://github.com/wandb/wandb) to provide feedback.
{{% /alert %}}

{{% alert title="Requirements" %}}
To track multiple processes to a single run, you must have:
- W&B Python SDK version `v0.19.9` or newer.

- W&B Server v0.68 or newer.
{{% /alert  %}}

In this approach you use a primary node and one or more worker nodes. Within the primary node you initialize a W&B run. For each worker node, initialize a run using the run ID used by the primary node. During training each worker node logs to the same run ID as the primary node. W&B aggregates metrics from all nodes and displays them in the W&B App UI.

Within the primary node, initialize a W&B run with [`wandb.init()`]({{< relref "/ref/python/sdk/functions/init" >}}). Pass in a `wandb.Settings` object to the `settings` parameter (`wandb.init(settings=wandb.Settings()`) with the following:

1. The `mode` parameter set to `"shared"` to enable shared mode.
2. A unique label for [`x_label`](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_settings.py#L638). You use the value you specify for `x_label` to identify which node the data is coming from in logs and system metrics in the W&B App UI. If left unspecified, W&B creates a label for you using the hostname and a random hash.
3. Set the [`x_primary`](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_settings.py#L660) parameter to `True` to indicate that this is the primary node.
4. Optionally provide a list of GPU indexes ([0,1,2]) to `x_stats_gpu_device_ids` to specify which GPUs W&B tracks metrics for. If you do not provide a list, W&B tracks metrics for all GPUs on the machine.

Make note of the run ID of the primary node. Each worker node needs the run ID of the primary node.

{{% alert %}}
`x_primary=True` distinguishes a primary node from worker nodes. Primary nodes are the only nodes that upload files shared across nodes such as configuration files, telemetry and more. Worker nodes do not upload these files.
{{% /alert %}}

For each worker node, initialize a W&B run with [`wandb.init()`]({{< relref "/ref/python/sdk/functions/init" >}}) and provide the following:
1. A `wandb.Settings` object to the `settings` parameter (`wandb.init(settings=wandb.Settings()`) with:
   * The `mode` parameter set to `"shared"` to enable shared mode.
   * A unique label for `x_label`. You use the value you specify for `x_label` to identify which node the data is coming from in logs and system metrics in the W&B App UI. If left unspecified, W&B creates a label for you using the hostname and a random hash.
   * Set the `x_primary` parameter to `False` to indicate that this is a worker node.
2. Pass the run ID used by the primary node to the `id` parameter.
3. Optionally set [`x_update_finish_state`](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_settings.py#L772) to `False`. This prevents non-primary nodes from updating the [run's state]({{< relref "/guides/models/track/runs/#run-states" >}}) to `finished` prematurely, ensuring the run state remains consistent and managed by the primary node.

{{% alert %}}
Consider using an environment variable to set the run ID of the primary node that you can then define in each worker node's machine.
{{% /alert %}}

The following sample code demonstrates the high level requirements for tracking multiple processes to a single run:

```python
import wandb

# Initialize a run in the primary node
run = wandb.init(
    entity="entity",
    project="project",
	settings=wandb.Settings(
        x_label="rank_0", 
        mode="shared", 
        x_primary=True,
        x_stats_gpu_device_ids=[0, 1],  # (Optional) Only track metrics for GPU 0 and 1
        )
)

# Note the run ID of the primary node.
# Each worker node needs this run ID.
run_id = run.id

# Initialize a run in a worker node using the run ID of the primary node
run = wandb.init(
	settings=wandb.Settings(x_label="rank_1", mode="shared", x_primary=False),
	id=run_id,
)

# Initialize a run in a worker node using the run ID of the primary node
run = wandb.init(
	settings=wandb.Settings(x_label="rank_2", mode="shared", x_primary=False),
	id=run_id,
)
```

In a real world example, each worker node might be on a separate machine.

{{% alert %}}
See the [Distributed Training with Shared Mode](https://wandb.ai/dimaduev/simple-cnn-ddp/reports/Distributed-Training-with-Shared-Mode--VmlldzoxMTI0NTE1NA) report for an end-to-end example on how to train a model on a multi-node and multi-GPU Kubernetes cluster in GKE.
{{% /alert %}}

View console logs from multi node processes in the project that the run logs to:

1. Navigate to the project that contains the run.
2. Click on the **Runs** tab in the left sidebar.
3. Click on the run you want to view.
4. Click on the **Logs** tab in the left sidebar.

You can filter console logs based on the labels you provide for `x_label` in the UI search bar located at the top of the console log page. For example, the following image shows which options are available to filter the console log by if values  `rank0`, `rank1`, `rank2`, `rank3`, `rank4`, `rank5`, and `rank6` are provided to `x_label`.` 

{{< img src="/images/track/multi_node_console_logs.png" alt="Multi-node console logs" >}}

See [Console logs]({{< relref "/guides/models/app/console-logs/" >}}) for more information.

W&B aggregates system metrics from all nodes and displays them in the W&B App UI. For example, the following image shows a sample dashboard with system metrics from multiple nodes. Each node possesses a unique label (`rank_0`, `rank_1`, `rank_2`) that you specify in the `x_label` parameter.

{{< img src="/images/track/multi_node_system_metrics.png" alt="Multi-node system metrics" >}}

See [Line plots]({{< relref "/guides/models/app/features/panels/line-plot/" >}}) for information on how to customize line plot panels. 

## Example use cases

The following code snippets demonstrate common scenarios for advanced distributed use cases.

### Spawn process

Use the `wandb.setup()`method in your main function if you initiate a run in a spawned process:

```python
import multiprocessing as mp

def do_work(n):
    with wandb.init(config=dict(n=n)) as run:
        run.log(dict(this=n * n))

def main():
    wandb.setup()
    pool = mp.Pool(processes=4)
    pool.map(do_work, range(4))


if __name__ == "__main__":
    main()
```

### Share a run

Pass a run object as an argument to share runs between processes:

```python
def do_work(run):
    with wandb.init() as run:
        run.log(dict(this=1))

def main():
    run = wandb.init()
    p = mp.Process(target=do_work, kwargs=dict(run=run))
    p.start()
    p.join()
    run.finish()  # mark the run as finished


if __name__ == "__main__":
    main()
```

W&B can not guarantee the logging order. Synchronization should be done by the author of the script.


## Troubleshooting

There are two common issues you might encounter when using W&B and distributed training:

1. **Hanging at the beginning of training** - A `wandb` process can hang if the `wandb` multiprocessing interferes with the multiprocessing from distributed training.
2. **Hanging at the end of training** - A training job might hang if the `wandb` process does not know when it needs to exit. Call the `wandb.Run.finish()` API at the end of your Python script to tell W&B that the run finished. The `wandb.Run.finish()` API will finish uploading data and will cause W&B to exit.
W&B recommends using `wandb service` command to improve the reliability of your distributed jobs. Both of the preceding training issues are commonly found in versions of the W&B SDK where wandb service is unavailable.

### Enable W&B Service

Depending on your version of the W&B SDK, you might already have W&B Service enabled by default.

#### W&B SDK 0.13.0 and above

W&B Service is enabled by default for versions of the W&B SDK `0.13.0` and above.

#### W&B SDK 0.12.5 and above

Modify your Python script to enable W&B Service for W&B SDK version 0.12.5 and above. Use the `wandb.require` method and pass the string `"service"` within your main function:

```python
if __name__ == "__main__":
    main()


def main():
    wandb.require("service")
    # rest-of-your-script-goes-here
```

For optimal experience we do recommend you upgrade to the latest version.

**W&B SDK 0.12.4 and below**

Set the `WANDB_START_METHOD` environment variable to `"thread"` to use multithreading instead if you use a W&B SDK version 0.12.4 and below.
