---
description: Sweeps quickstart shows how to define, initialize, and run a sweep. There
  are four main steps
menu:
  default:
    identifier: ja-guides-models-sweeps-walkthrough
    parent: sweeps
title: 'Tutorial: Define, initialize, and run a sweep'
weight: 1
---

This page shows how to define, initialize, and run a sweep. There are four main steps:

1. [Set up your training code]({{< relref path="#set-up-your-training-code" lang="ja" >}})
2. [Define the search space with a sweep configuration]({{< relref path="#define-the-search-space-with-a-sweep-configuration" lang="ja" >}})
3. [Initialize the sweep]({{< relref path="#initialize-the-sweep" lang="ja" >}})
4. [Start the sweep agent]({{< relref path="#start-the-sweep" lang="ja" >}})


Copy and paste the following code into a Jupyter Notebook or Python script:

```python 
# Import the W&B Python Library and log into W&B
import wandb

wandb.login()

# 1: Define objective/training function
def objective(config):
    score = config.x**3 + config.y
    return score

def main():
    wandb.init(project="my-first-sweep")
    score = objective(wandb.config)
    wandb.log({"score": score})

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
Define a training function that takes in hyperparameter values from `wandb.config` and uses them to train a model and return metrics.

Optionally provide the name of the project where you want the output of the W&B Run to be stored (project parameter in [`wandb.init`]({{< relref path="/ref/python/init.md" lang="ja" >}})). If the project is not specified, the run is put in an "Uncategorized" project.

{{% alert %}}
Both the sweep and the run must be in the same project. Therefore, the name you provide when you initialize W&B must match the name of the project you provide when you initialize a sweep.
{{% /alert %}}

```python
# 1: Define objective/training function
def objective(config):
    score = config.x**3 + config.y
    return score


def main():
    wandb.init(project="my-first-sweep")
    score = objective(wandb.config)
    wandb.log({"score": score})
```

## Define the search space with a sweep configuration
Within a dictionary, specify what hyperparameters you want to sweep over and. For more information about configuration options, see [Define sweep configuration]({{< relref path="/guides/models/sweeps/define-sweep-configuration/" lang="ja" >}}).

The proceeding example demonstrates a sweep configuration that uses a random search (`'method':'random'`). The sweep will randomly select a random set of values listed in the configuration for the batch size, epoch, and the learning rate.

Throughout the sweeps, W&B will maximize the metric specified in the metric key (`metric`). In the following example, W&B will maximize (`'goal':'maximize'`) the validation accuracy (`'val_acc'`).


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

W&B uses a _Sweep Controller_ to manage sweeps on the cloud (standard), locally (local) across one or more machines. For more information about Sweep Controllers, see [Search and stop algorithms locally]({{< relref path="./local-controller.md" lang="ja" >}}).

A sweep identification number is returned when you initialize a sweep:

```python
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")
```

For more information about initializing sweeps, see [Initialize sweeps]({{< relref path="./initialize-sweeps.md" lang="ja" >}}).

## Start the Sweep

Use the [`wandb.agent`]({{< relref path="/ref/python/agent.md" lang="ja" >}}) API call to start a sweep.

```python
wandb.agent(sweep_id, function=main, count=10)
```

## Visualize results (optional)

Open your project to see your live results in the W&B App dashboard. With just a few clicks, construct rich, interactive charts like [parallel coordinates plots]({{< relref path="/guides/models/app/features/panels/parallel-coordinates.md" lang="ja" >}}),[ parameter importance analyzes]({{< relref path="/guides/models/app/features/panels/parameter-importance.md" lang="ja" >}}), and [more]({{< relref path="/guides/models/app/features/panels/" lang="ja" >}}).

{{< img src="/images/sweeps/quickstart_dashboard_example.png" alt="Sweeps Dashboard example" >}}

For more information about how to visualize results, see [Visualize sweep results]({{< relref path="./visualize-sweep-results.md" lang="ja" >}}). For an example dashboard, see this sample [Sweeps Project](https://wandb.ai/anmolmann/pytorch-cnn-fashion/sweeps/pmqye6u3).

## Stop the agent (optional)

From the terminal, hit `Ctrl+c` to stop the run that the Sweep agent is currently running. To kill the agent, hit `Ctrl+c` again after the run is stopped.