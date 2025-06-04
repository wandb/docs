---
menu:
  tutorials:
    identifier: sweeps
    parent: null
title: Tune hyperparameters with sweeps
weight: 3
---
Finding a machine learning model to achieve a specific metric, like accuracy, often requires many iterations. Choosing hyperparameter combinations for training can remain unclear. Use W&B Sweeps to automatically find optimal hyperparameter values, like learning rate, batch size, number of hidden layers, and optimizer type to meet your metric.

In this tutorial, create a hyperparameter search using the W&B PyTorch integration. Follow along with a [video tutorial](http://wandb.me/sweeps-video).

## Sweeps: An overview

Running a hyperparameter sweep with Weights & Biases consists of three steps:

1. **Define the sweep:** Create a dictionary or a [YAML file]({{< relref "/guides/models/sweeps/define-sweep-configuration" >}}) specifying the parameters to search, the search strategy, and the optimization metric.
2. **Initialize the sweep:** Use one line of code to initialize the sweep with the configuration dictionary: `sweep_id = wandb.sweep(sweep_config)`.
3. **Run the sweep personal digital assistant:** Call `wandb.agent()` with the `sweep_id` to run it with a function that defines the model architecture and trains it: `wandb.agent(sweep_id, function=train)`.

## Before you start

Install W&B and import the W&B Python SDK in your notebook:

1. Install W&B:
```
!pip install wandb -Uq
```

2. Import W&B:
```
import wandb
```

3. Log in to W&B by entering your API key when prompted:
```
wandb.login()
```

## Step 1: Define a sweep

A W&B Sweep combines a strategy for testing hyperparameter values with code that evaluates them. Define your sweep strategy with a _sweep configuration_ before starting the sweep.

{{% alert %}}
For sweeps in a Jupyter Notebook, the sweep configuration must be in a nested dictionary. For command-line sweeps, use a [YAML file]({{< relref "/guides/models/sweeps/define-sweep-configuration" >}}).
{{% /alert %}}

### Pick a search method

Specify a hyperparameter search method in your configuration dictionary. Choose from grid, random, and Bayesian search. Use random search in this tutorial. Create a dictionary in your notebook and set `method` to `random`.

```
sweep_config = {
    'method': 'random'
}
```

Specify a metric to optimize. Although not required for random search, it's good practice.

```
metric = {
    'name': 'loss',
    'goal': 'minimize'
}

sweep_config['metric'] = metric
```

### Specify hyperparameters to search

After setting a search method, specify the hyperparameters to search. Assign hyperparameter names to `parameter` and their possible values to `value`. Values depend on the hyperparameter type.

For example, when choosing a machine learning optimizer, specify optimizer names like Adam and stochastic gradient descent.

```
parameters_dict = {
    'optimizer': {
        'values': ['adam', 'sgd']
    },
    'fc_layer_size': {
        'values': [128, 256, 512]
    },
    'dropout': {
        'values': [0.3, 0.4, 0.5]
    },
}

sweep_config['parameters'] = parameters_dict
```

To track a hyperparameter without varying its value, add it to the sweep configuration with a specific value. In the example below, `epochs` is set to 1.

```
parameters_dict.update({
    'epochs': {
        'value': 1
    }
})
```

In a `random` search, all parameter values have equal chances of selection. To specify a `distribution` and its parameters, use `mu` and `sigma` for a `normal` distribution.

```
parameters_dict.update({
    'learning_rate': {
        'distribution': 'uniform',
        'min': 0,
        'max': 0.1
    },
    'batch_size': {
        'distribution': 'q_log_uniform_values',
        'q': 8,
        'min': 32,
        'max': 256
    }
})
```

When done, `sweep_config` specifies the `parameters` to explore and the `method` to use.

Examine the sweep configuration:

```
import pprint
pprint.pprint(sweep_config)
```

For a full list of configuration options, see [Sweep configuration options]({{< relref "/guides/models/sweeps/define-sweep-configuration/sweep-config-keys/" >}}).

{{% alert %}}
When hyperparameters have infinite options, try a few select `values`. In the example above, finite values are specified for `layer_size` and `dropout`.
{{% /alert %}}

## Step 2: Initialize the sweep

After defining your search strategy, set up its implementation.

W&B uses a Sweep Controller to manage sweeps on the cloud or across machines. Use a W&B-managed sweep controller for this tutorial. While controllers manage sweeps, a _sweep personal digital assistant_ runs them.

{{% alert %}}
By default, sweep controllers operate on W&B's servers, and sweep agents run on your local machine.
{{% /alert %}}

Activate a sweep controller in your notebook using `wandb.sweep`. Pass your sweep configuration dictionary to it:

```
sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")
```

The `wandb.sweep` function returns a `sweep_id`, which you use later to activate your sweep.

{{% alert %}}
On the command line, use:

```python
wandb sweep config.yaml
```
{{% /alert %}}

For more details on terminal-based Sweeps, see the [W&B Sweep walkthrough]({{< relref "/guides/models/sweeps/walkthrough" >}}).

## Step 3: Define your machine learning code

Before running the sweep, define the training procedure using the hyperparameter values. Make sure each training experiment's logic can access the hyperparameter values in your sweep configuration.

In the example below, `build_dataset`, `build_network`, `build_optimizer`, and `train_epoch` access the sweep hyperparameter configuration dictionary.

Run the machine learning training code in your notebook. These functions define a basic fully connected neural network using PyTorch.

```python
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(config=None):
    with wandb.init(config=config):
        config = wandb.config

        loader = build_dataset(config.batch_size)
        network = build_network(config.fc_layer_size, config.dropout)
        optimizer = build_optimizer(network, config.optimizer, config.learning_rate)

        for epoch in range(config.epochs):
            avg_loss = train_epoch(network, loader, optimizer)
            wandb.log({"loss": avg_loss, "epoch": epoch})           
```

In the `train` function, note these W&B Python SDK methods:
* [`wandb.init()`]({{< relref "/ref/python/init" >}}): Initializes a new W&B run. Each run is a training function execution.
* [`wandb.config`]({{< relref "/guides/models/track/config" >}}): Passes the sweep configuration and hyperparameters to experiment with.
* [`wandb.log()`]({{< relref "/ref/python/log" >}}): Logs the training loss by epoch.

This cell defines the functions `build_dataset`, `build_network`, `build_optimizer`, and `train_epoch`. These functions are part of a basic PyTorch pipeline, unaffected by W&B.

```python
def build_dataset(batch_size):
   
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST(".", train=True, download=True,
                             transform=transform)
    sub_dataset = torch.utils.data.Subset(
        dataset, indices=range(0, len(dataset), 5))
    loader = torch.utils.data.DataLoader(sub_dataset, batch_size=batch_size)

    return loader


def build_network(fc_layer_size, dropout):
    network = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, fc_layer_size), nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(fc_layer_size, 10),
        nn.LogSoftmax(dim=1))

    return network.to(device)
        

def build_optimizer(network, optimizer, learning_rate):
    optimizer = optim.SGD if optimizer == "sgd" else optim.Adam
    return optimizer(network.parameters(), lr=learning_rate, momentum=0.9 if optimizer == optim.SGD else None)


def train_epoch(network, loader, optimizer):
    cumu_loss = 0
    for _, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        loss = F.nll_loss(network(data), target)
        cumu_loss += loss.item()

        loss.backward()
        optimizer.step()

        wandb.log({"batch loss": loss.item()})

    return cumu_loss / len(loader)
```

For more details on using W&B with PyTorch, see [this Colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb).

## Step 4: Activate sweep agents

With your sweep configuration and training script defined, activate a sweep agent. Sweep agents run experiments with the defined hyperparameter sets.

Create sweep agents with `wandb.agent`. Provide:
1. The sweep the agent is part of (`sweep_id`)
2. The function the sweep runs. Here, the sweep uses the `train` function.
3. Optionally, the number of configs to request from the sweep controller (`count`).

{{% alert %}}
Start multiple sweep agents with the same `sweep_id` on different compute resources. The sweep controller ensures they work together according to your sweep configuration.
{{% /alert %}}

The following cell activates a sweep agent that runs the training function (`train`) five times:

```python
wandb.agent(sweep_id, train, count=5)
```

{{% alert %}}
With the `random` search method specified, the sweep controller provides random hyperparameter values.
{{% /alert %}}

For more details on terminal-based Sweeps, see the [W&B Sweep walkthrough]({{< relref "/guides/models/sweeps/walkthrough" >}}).

## Visualize sweep results

### Parallel coordinates plot

This plot maps hyperparameter values to model metrics and helps identify hyperparameter combinations that improve model performance.

{{< img src="/images/tutorials/sweeps-2.png" alt="" >}}

### Hyperparameter importance plot

This plot identifies which hyperparameters best predict your metrics. It reports feature importance (from a random forest model) and correlation (implicitly a linear model).

{{< img src="/images/tutorials/sweeps-3.png" alt="" >}}

These visualizations help save time and resources by identifying critical parameters and value ranges for further exploration.

## Learn more about W&B sweeps

Explore a basic training script and [a variety of sweep configurations](https://github.com/wandb/examples/tree/master/examples/keras/keras-cnn-fashion).

The repository also includes advanced sweep features like [Bayesian Hyperband](https://app.wandb.ai/wandb/examples-keras-cnn-fashion/sweeps/us0ifmrf?workspace=user-lavanyashukla) and [Hyperopt](https://app.wandb.ai/wandb/examples-keras-cnn-fashion/sweeps/xbs2wm5e?workspace=user-lavanyashukla).