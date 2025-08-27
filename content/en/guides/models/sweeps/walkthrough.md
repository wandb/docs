---
description: Sweeps quickstart shows how to define, initialize, and run a sweep. There
  are four main steps
menu:
  default:
    identifier: walkthrough_sweeps
    parent: sweeps
title: 'Tutorial: Define, initialize, and run a sweep'
weight: 1
---

This page shows how to define, initialize, and run a sweep. There are four main steps:

1. [Set up your training code]({{< relref "#set-up-your-training-code" >}})
2. [Define the search space with a sweep configuration]({{< relref "#define-the-search-space-with-a-sweep-configuration" >}})
3. [Initialize the sweep]({{< relref "#initialize-the-sweep" >}})
4. [Start the sweep agent]({{< relref "#start-the-sweep" >}})


Copy and paste the following code into a Jupyter Notebook or Python script:

```python
# Import the W&B Python Library and log into W&B
import wandb

# 1: Define objective/training function
def objective(config):
    score = config.x**3 + config.y
    return score

def main():
    with wandb.init(project="my-first-sweep") as run:
        score = objective(run.config)
        run.log({"score": score})

# 2: Define the search space
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "score"},
    "parameters": {
        "x": {"max": 0.1, "min": 0.01},
        "y": {"values": [1, 3, 7]},
    },
}

# 3: Start the sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")

wandb.agent(sweep_id, function=main, count=10)
```

The following sections break down and explains each step in the code sample.


## Set up your training code
Define a training function that takes in hyperparameter values from `wandb.Run.config` and uses them to train a model and return metrics.

Optionally provide the name of the project where you want the output of the W&B Run to be stored (project parameter in [`wandb.init()`]({{< relref "/ref/python/sdk/functions/init.md" >}})). If the project is not specified, the run is put in an "Uncategorized" project.

{{% alert %}}
Both the sweep and the run must be in the same project. Therefore, the name you provide when you initialize W&B must match the name of the project you provide when you initialize a sweep.
{{% /alert %}}

```python
# 1: Define objective/training function
def objective(config):
    score = config.x**3 + config.y
    return score


def main():
    with wandb.init(project="my-first-sweep") as run:
        score = objective(run.config)
        run.log({"score": score})
```

## Define the search space with a sweep configuration

Specify the hyperparameters to sweep in a dictionary. For configuration options, see [Define sweep configuration]({{< relref "/guides/models/sweeps/define-sweep-configuration/" >}}).

The proceeding example demonstrates a sweep configuration that uses a random search (`'method':'random'`). The sweep will randomly select a random set of values listed in the configuration for the batch size, epoch, and the learning rate.

W&B minimizes the metric specified in the `metric` key when `"goal": "minimize"` is associated with it. In this case, W&B will optimize for minimizing the metric  `score` (`"name": "score"`).


```python
# 2: Define the search space
sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "score"},
    "parameters": {
        "x": {"max": 0.1, "min": 0.01},
        "y": {"values": [1, 3, 7]},
    },
}
```

## Initialize the Sweep

W&B uses a _Sweep Controller_ to manage sweeps on the cloud (standard), locally (local) across one or more machines. For more information about Sweep Controllers, see [Search and stop algorithms locally]({{< relref "./local-controller.md" >}}).

A sweep identification number is returned when you initialize a sweep:

```python
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")
```

For more information about initializing sweeps, see [Initialize sweeps]({{< relref "./initialize-sweeps.md" >}}).

## Start the Sweep

Use the [`wandb.agent()`]({{< relref "/ref/python/sdk/functions/agent.md" >}}) API call to start a sweep.

```python
wandb.agent(sweep_id, function=main, count=10)
```

{{% alert color="secondary" title="Multiprocessing" %}}
You must wrap your `wandb.agent()` and `wandb.sweep()` calls with `if __name__ == '__main__':` if you use Python standard library's `multiprocessing` or PyTorch's `pytorch.multiprocessing` package. For example:

```python
if __name__ == '__main__':
    wandb.agent(sweep_id="<sweep_id>", function="<function>", count="<count>")
```

Wrapping your code with this convention ensures that it is only executed when the script is run directly, and not when it is imported as a module in a worker process.

See [Python standard library `multiprocessing`](https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods) or [PyTorch `multiprocessing`](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html#asynchronous-multiprocess-training-e-g-hogwild) for more information about multiprocessing. See https://realpython.com/if-name-main-python/ for information about the `if __name__ == '__main__':` convention.
{{% /alert %}}


## Visualize results (optional)

Open your project to see your live results in the W&B App dashboard. With just a few clicks, construct rich, interactive charts like [parallel coordinates plots]({{< relref "/guides/models/app/features/panels/parallel-coordinates.md" >}}),[ parameter importance analyzes]({{< relref "/guides/models/app/features/panels/parameter-importance.md" >}}), and [additional chart types]({{< relref "/guides/models/app/features/panels/" >}}).

{{< img src="/images/sweeps/quickstart_dashboard_example.png" alt="Sweeps Dashboard example" >}}

For more information about how to visualize results, see [Visualize sweep results]({{< relref "./visualize-sweep-results.md" >}}). For an example dashboard, see this sample [Sweeps Project](https://wandb.ai/anmolmann/pytorch-cnn-fashion/sweeps/pmqye6u3).

## Stop the agent (optional)

In the terminal, press `Ctrl+C` to stop the current run. Press it again to terminate the agent.
