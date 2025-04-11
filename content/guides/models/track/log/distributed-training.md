---
description: Use W&B to log distributed training experiments with multiple GPUs.
menu:
  default:
    identifier: distributed-training
    parent: log-objects-and-media
title: Log distributed training experiments
---

During distributed training you train models using multiple machines or clients in parallel. You can use W&B to track distributed training experiments. Based on your use case, use one of the following methods to track distributed training experiments:

* **Single process**: Track a rank 0 (also known as a "leader" or "coordinator") process with W&B. This is a common solution for logging distributed training experiments with the [PyTorch Distributed Data Parallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel) (DDP) Class. 
* **Multiple processes**: For multiple processes, you can either:
   * Track each process separately using a different run for each process. Optionally group them together in the W&B App UI.
   * Track all process to a single run.

<!-- The proceeding examples demonstrate how to track metrics with W&B using PyTorch DDP on two GPUs on a single machine. [PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) (`DistributedDataParallel` in`torch.nn`) is a popular library for distributed training. The basic principles apply to any distributed training setup, but the details of implementation may differ.

{{% alert %}}
Explore the code behind these examples in the W&B GitHub examples repository [here](https://github.com/wandb/examples/tree/master/examples/pytorch/pytorch-ddp). Specifically, see the [`log-dpp.py`](https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py) Python script for information on how to implement one process and many process methods.
{{% /alert %}} -->

## Single process

In this method you track only values and metrics available to your rank 0 process. Within the rank 0 process, initialize a W&B run with [`wandb.init`]({{< relref "/ref//python/init.md" >}}) and log experiments ([`wandb.log`]({{< relref "/ref//python/log.md" >}})) to that run.

Metrics from other nodes or processes, such as loss values or inputs from training batches, are not tracked. System metrics, such as usage and memory, are logged for all GPUs since that information is available to all processes.

{{% alert %}}
**Use this method to only track metrics available from a single process**. Typical examples include GPU/CPU utilization, behavior on a shared validation set, gradients and parameters, and loss values on representative data examples.
{{% /alert %}}

As an example, the following [sample Python script (`log-ddp.py`)](https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py) demonstrate how to track metrics on two GPUs on a single machine using PyTorch DDP. [PyTorch DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) (`DistributedDataParallel` in`torch.nn`) is a popular library for distributed training. The basic principles apply to any distributed training setup, but the details of implementation may differ.

The Python script first start multiple processes with `torch.distributed.launch`. Next, it checks the rank with the `--local_rank` command line argument. If the rank is set to 0, set up `wandb` logging conditionally in the [`train()`](https://github.com/wandb/examples/blob/master/examples/pytorch/pytorch-ddp/log-ddp.py#L24) function.

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

Explore this example dashboard in the W&B App UI to [view metrics tracked from a single process](https://wandb.ai/ayush-thakur/DDP/runs/1s56u3hc/system). The dashboard displays system metrics such as temperature and utilization, that were tracked for both GPUs.

{{< img src="/images/track/distributed_training_method1.png" alt="" >}}

However, the loss values as a function epoch and batch size were only logged from a single GPU.

{{< img src="/images/experiments/loss_function_single_gpu.png" alt="" >}}

## Multiple processes

### Track each process separately

In this approach you track each process separately by creating a run for each process. You then log metrics, artifacts, and forth to their respective run. Call `wandb.finish()` at the end of training, to mark that the run has completed so that all processes exit properly.

You might find it difficult to keep track of runs across multiple experiments. To mitigate this, provide a value to the `group` parameter when you initialize W&B (`wandb.init(group='group-name')`) to keep track of which run belongs to a given experiment. For more information about how to keep track of training and evaluation W&B Runs in experiments, see [Group Runs]({{< relref "/guides/models/track/runs/grouping.md" >}}).

{{% alert %}}
**Use this method if you want to track metrics from individual processes**. Typical examples include the data and predictions on each node (for debugging data distribution) and metrics on individual batches outside of the main node. This method is not necessary to get system metrics from all nodes nor to get summary statistics available on the main node.
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
```

Explore the W&B App UI to view an [example dashboard](https://wandb.ai/ayush-thakur/DDP?workspace=user-noahluna) of metrics tracked from multiple processes. Note that there are two W&B Runs grouped together in the left sidebar. Click on a group to view the dedicated group page for the experiment. The dedicated group page displays metrics from each process separately.

{{< img src="/images/experiments/dashboard_grouped_runs.png" alt="" >}}

The preceding image demonstrates the W&B App UI dashboard. On the sidebar we see two experiments. One labeled 'null' and a second (bound by a yellow box) called 'DPP'. If you expand the group (select the Group dropdown) you will see the W&B Runs that are associated to that experiment.

### Track all processes to a single run

{{% alert color="secondary"  %}}
The option to track multiple processes to a single run is in public preview.
{{% /alert %}}

{{% alert %}}
To track multiple processes to a single run, you must have W&B Python SDK version `>=0.19.0`.
{{% /alert  %}}

In this approach you use a primary node and one or more worker nodes. Within the primary node you initialize a W&B run. For each worker node, initialize a run using the run ID used by the primary node. During training each worker node logs to the same run ID as the primary node. W&B aggregates metrics from all nodes and displays them in the W&B App UI.

Within the primary node, initialize a W&B run with [`wandb.init`]({{< relref "/ref/python/init.md" >}}) and provide the following:

1. A `wandb.Settings` object to the `settings` parameter (`wandb.init(settings=wandb.Settings()`) with:
   * The `mode` parameter set to `"shared"` to enable shared mode.
   * A unique label for `x_label`. You use the value you specify for `x_label` to identify which process the data is coming from in logs and system metrics in the W&B App UI.
2. Set the `x_primary` parameter to `True` to indicate that this is the primary node.

Make note of the run ID of the primary node. Each worker node needs this run ID.

For each worker node, initialize a W&B run with [`wandb.init`]({{< relref "/ref/python/init.md" >}}) and provide the following:
1. A `wandb.Settings` object to the `settings` parameter (`wandb.init(settings=wandb.Settings()`) with:
   * The `mode` parameter set to `"shared"` to enable shared mode.
   * A unique label for `x_label`. You use the value you specify for `x_label` to identify which process the data is coming from in logs and system metrics in the W&B App UI.
2. Set the `x_primary` parameter to `False` to indicate that this is a worker node.
3. Pass the run ID of the primary node to the `id` parameter.
4. Optionally set `x_update_finish_state` to `False`. This prevents non-primary nodes from updating the [run's state]({{< relref "/guides/models/track/runs/#run-states" >}}) to `finished` prematurely, ensuring the run state remains consistent and managed by the primary node.

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
	settings=wandb.Settings(x_label="rank_0", mode="shared"),
    x_primary=True,
)

# Note the run ID of the primary node.
# Each worker node needs this run ID.
run_id = run.id

# Initialize a run in a worker node using the run ID of the primary node
run = wandb.init(
	settings=wandb.Settings(x_label="rank_1", mode="shared"),
	id=run_id,
    x_primary=False,
)

# Initialize a run in a worker node using the run ID of the primary node
run = wandb.init(
	settings=wandb.Settings(x_label="rank_2", mode="shared"),
	id=run_id,
    x_primary=True,
)
```

{{% alert %}}
See this report for an end-to-end example on how to [train a model on a multi-node multi-GPU Kubernetes cluster in GKE](https://wandb.ai/dimaduev/simple-cnn-ddp/reports/Distributed-Training-with-Shared-Mode--VmlldzoxMTI0NTE1NA). 
{{% /alert %}}

View console logs from multi node processes in the project that the run logs to:

1. Navigate to the project that contains the run.
2. Click on the **Runs** tab in the left sidebar.
3. Click on the run you want to view.
4. Click on the **Logs** tab in the left sidebar.

System metrics within your project's workspace aggregate metrics from all nodes and display them in the W&B App UI. For example, the following image shows a sample dashboard with system metrics from multiple nodes. Each node possesses a unique label (`rank_0`, `rank_1`, `rank_2`) that you specified in the `x_label` parameter.

{{< img src="/images/track/multi_node_system_metrics.png" alt="Line plot panel with numerous semi linear lines depicting system metrics logged by a multi node process." >}}

See [Line plots]({{< relref "/guides/models/app/features/panels/line-plot/" >}}) for information on how to customize line plot panels. 

## Avoid common distributed training issues with W&B Service

There are two common issues you might encounter when using W&B and distributed training:

1. **Hanging at the beginning of training** - A `wandb` process can hang if the `wandb` multiprocessing interferes with the multiprocessing from distributed training.
2. **Hanging at the end of training** - A training job might hang if the `wandb` process does not know when it needs to exit. Call the `wandb.finish()` API at the end of your Python script to tell W&B that the Run finished. The wandb.finish() API will finish uploading data and will cause W&B to exit.

We recommend using the `wandb service` to improve the reliability of your distributed jobs. Both of the preceding training issues are commonly found in versions of the W&B SDK where wandb service is unavailable.

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

## Example use cases for multiprocessing

The following code snippets demonstrate common methods for advanced distributed use cases.

### Spawn process

Use the `wandb.setup()`method in your main function if you initiate a W&B Run in a spawned process:

```python
import multiprocessing as mp


def do_work(n):
    run = wandb.init(config=dict(n=n))
    run.log(dict(this=n * n))


def main():
    wandb.setup()
    pool = mp.Pool(processes=4)
    pool.map(do_work, range(4))


if __name__ == "__main__":
    main()
```

### Share a run

Pass a W&B run object as an argument to share runs between processes:

```python
def do_work(run):
    run.log(dict(this=1))


def main():
    run = wandb.init()
    p = mp.Process(target=do_work, kwargs=dict(run=run))
    p.start()
    p.join()


if __name__ == "__main__":
    main()
```


{{% alert %}}
Note that we can not guarantee the logging order. Synchronization should be done by the author of the script.
{{% /alert %}}