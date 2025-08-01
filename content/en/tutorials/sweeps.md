---
menu:
  tutorials:
    identifier: sweeps
    parent: null
title: Tune hyperparameters with sweeps
weight: 3
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W&B.ipynb" >}}

Finding a machine learning model that meets your desired metric (such as model accuracy) is normally a redundant task that can take multiple iterations. To make matters worse, it might be unclear which hyperparameter combinations to use for a given training run. 

Use W&B Sweeps to create an organized and efficient way to automatically search through combinations of hyperparameter values such as the learning rate, batch size, number of hidden layers, optimizer type and more to find values that optimize your model based on your desired metric.

In this tutorial you will create a hyperparameter search with W&B PyTorch integration. Follow along with a [video tutorial](https://wandb.me/sweeps-video).

{{< img src="/images/tutorials/sweeps-1.png" alt="Hyperparameter sweep results" >}}

## Sweeps: An Overview

Running a hyperparameter sweep with Weights & Biases is very easy. There are just 3 simple steps:

1. **Define the sweep:** we do this by creating a dictionary or a [YAML file]({{< relref "/guides/models/sweeps/define-sweep-configuration" >}}) that specifies the parameters to search through, the search strategy, the optimization metric et all.

2. **Initialize the sweep:** with one line of code we initialize the sweep and pass in the dictionary of sweep configurations:
`sweep_id = wandb.sweep(sweep_config)`

3. **Run the sweep agent:** also accomplished with one line of code, we call `wandb.agent()` and pass the `sweep_id` to run, along with a function that defines your model architecture and trains it:
`wandb.agent(sweep_id, function=train)`


## Before you get started

Install W&B and import the W&B Python SDK into your notebook:

1. Install with `!pip install`:


```
!pip install wandb -Uq
```

2. Import W&B:


```
import wandb
```

3. Log in to W&B and provide your API key when prompted:


```
wandb.login()
```

## Step 1️: Define a sweep

A W&B Sweep combines a strategy for trying numerous hyperparameter values with the code that evaluates them.
Before you start a sweep, you must define your sweep strategy with a _sweep configuration_.


{{% alert %}}
The sweep configuration you create for a sweep must be in a nested dictionary if you start a sweep in a Jupyter Notebook.

If you run a sweep within the command line, you must specify your sweep config with a [YAML file]({{< relref "/guides/models/sweeps/define-sweep-configuration" >}}).
{{% /alert %}}

### Pick a search method

First, specify a hyperparameter search method within your configuration dictionary. [There are three hyperparameter search strategies to choose from: grid, random, and Bayesian search]({{< relref "/guides/models/sweeps/define-sweep-configuration/sweep-config-keys/#method" >}}).

For this tutorial, you will use a random search. Within your notebook, create a dictionary and specify `random` for the `method` key. 


```
sweep_config = {
    'method': 'random'
    }
```

Specify a metric that you want to optimize for. You do not need to specify the metric and goal for sweeps that use random search method. However, it is good practice to keep track of your sweep goals because you can refer to it at a later time.


```
metric = {
    'name': 'loss',
    'goal': 'minimize'   
    }

sweep_config['metric'] = metric
```

### Specify hyperparameters to search through

Now that you have a search method specified in your sweep configuration, specify the hyperparameters you want to search over.

To do this, specify one or more hyperparameter names to the `parameter` key and specify one or more hyperparameter values for the `value` key.

The values you search through for a given hyperparameter depend on the type of hyperparameter you are investigating.  

For example, if you choose a machine learning optimizer, you must specify one or more finite optimizer names such as the Adam optimizer and stochastic gradient dissent.


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

Sometimes you want to track a hyperparameter, but not vary its value. In this case, add the hyperparameter to your sweep configuration and specify the exact value that you want to use. For example, in the following code cell, `epochs` is set to 1.


```
parameters_dict.update({
    'epochs': {
        'value': 1}
    })
```

For a `random` search,
all the `values` of a parameter are equally likely to be chosen on a given run.

Alternatively,
you can specify a named `distribution`,
plus its parameters, like the mean `mu`
and standard deviation `sigma` of a `normal` distribution.


```
parameters_dict.update({
    'learning_rate': {
        # a flat distribution between 0 and 0.1
        'distribution': 'uniform',
        'min': 0,
        'max': 0.1
      },
    'batch_size': {
        # integers between 32 and 256
        # with evenly-distributed logarithms 
        'distribution': 'q_log_uniform_values',
        'q': 8,
        'min': 32,
        'max': 256,
      }
    })
```

When we're finished, `sweep_config` is a nested dictionary
that specifies exactly which `parameters` we're interested in trying
and the `method` we're going to use to try them.

Let's see how the sweep configuration looks like:


```
import pprint
pprint.pprint(sweep_config)
```

For a full list of configuration options, see [Sweep configuration options]({{< relref "/guides/models/sweeps/define-sweep-configuration/sweep-config-keys/" >}}). 

{{% alert %}}
For hyperparameters that have potentially infinite options,
it usually makes sense to try out
a few select `values`. For example, the preceding sweep configuration has a list of finite values specified for the `layer_size` and `dropout` parameter keys.
{{% /alert %}}

## Step 2️: Initialize the Sweep

Once you've defined the search strategy, it's time to set up something to implement it.

W&B uses a Sweep Controller to manage sweeps on the cloud or locally across one or more machines. For this tutorial, you will use a sweep controller managed by W&B.

While sweep controllers manage sweeps, the component that actually executes a sweep is known as a _sweep agent_.


{{% alert %}}
By default, sweep controllers components are initiated on W&B's servers and sweep agents, the component that creates sweeps, are activated on your local machine.
{{% /alert %}}


Within your notebook, you can activate a sweep controller with the `wandb.sweep` method. Pass your sweep configuration dictionary you defined earlier to the `sweep_config` field:


```
sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")
```

The `wandb.sweep` function returns a `sweep_id` that you will use at a later step to activate your sweep.

{{% alert %}}
On the command line, this function is replaced with
```python
wandb sweep config.yaml
```
{{% /alert %}}

For more information on how to create W&B Sweeps in a terminal, see the [W&B Sweep walkthrough]({{< relref "/guides/models/sweeps/walkthrough" >}}).


## Step 3: Define your machine learning code

Before you execute the sweep,
define the training procedure that uses the hyperparameter values you want to try. The key to integrating W&B Sweeps into your training code is to ensure that, for each training experiment, that your training logic can access the hyperparameter values you defined in your sweep configuration.

In the proceeding code example, the helper functions `build_dataset`, `build_network`, `build_optimizer`, and `train_epoch` access the sweep hyperparameter configuration dictionary. 

Run the proceeding machine learning training code in your notebook. The functions define a basic fully connected neural network in PyTorch.


```python
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config) as run:
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = run.config

        loader = build_dataset(config.batch_size)
        network = build_network(config.fc_layer_size, config.dropout)
        optimizer = build_optimizer(network, config.optimizer, config.learning_rate)

        for epoch in range(config.epochs):
            avg_loss = train_epoch(network, loader, optimizer)
            run.log({"loss": avg_loss, "epoch": epoch})           
```

Within the `train` function, you will notice the following W&B Python SDK methods:
* [`wandb.init()`]({{< relref "/ref/python/sdk/functions/init/" >}}): Initialize a new W&B run. Each run is a single execution of the training function.
* [`run.config`]({{< relref "/guides/models/track/config" >}}): Pass sweep configuration with the hyperparameters you want to experiment with.
* [`run.log()`]({{< relref "/ref/python/sdk/classes/run/#method-runlog" >}}): Log the training loss for each epoch.


The proceeding cell defines four functions:
`build_dataset`, `build_network`, `build_optimizer`, and `train_epoch`. 
These functions are a standard part of a basic PyTorch pipeline,
and their implementation is unaffected by the use of W&B.


```python
def build_dataset(batch_size):
   
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3081,))])
    # download MNIST training dataset
    dataset = datasets.MNIST(".", train=True, download=True,
                             transform=transform)
    sub_dataset = torch.utils.data.Subset(
        dataset, indices=range(0, len(dataset), 5))
    loader = torch.utils.data.DataLoader(sub_dataset, batch_size=batch_size)

    return loader


def build_network(fc_layer_size, dropout):
    network = nn.Sequential(  # fully connected, single hidden layer
        nn.Flatten(),
        nn.Linear(784, fc_layer_size), nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(fc_layer_size, 10),
        nn.LogSoftmax(dim=1))

    return network.to(device)
        

def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(),
                               lr=learning_rate)
    return optimizer


def train_epoch(network, loader, optimizer):
    cumu_loss = 0

    with wandb.init() as run:
        for _, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            # ➡ Forward pass
            loss = F.nll_loss(network(data), target)
            cumu_loss += loss.item()

            # ⬅ Backward pass + weight update
            loss.backward()
            optimizer.step()

            run.log({"batch loss": loss.item()})

    return cumu_loss / len(loader)
```

For more details on instrumenting W&B with PyTorch, see [this Colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Simple_PyTorch_Integration.ipynb).

## Step 4: Activate sweep agents
Now that you have your sweep configuration defined and a training script that can utilize those hyperparameter in an interactive way, you are ready to activate a sweep agent. Sweep agents are responsible for running an experiment with a set of hyperparameter values that you defined in your sweep configuration.

Create sweep agents with the `wandb.agent` method. Provide the following:
1. The sweep the agent is a part of (`sweep_id`)
2. The function the sweep is supposed to run. In this example, the sweep will use the `train` function.
3. (optionally) How many configs to ask the sweep controller for (`count`)

{{% alert %}}
You can start multiple sweep agents with the same `sweep_id`
on different compute resources. The sweep controller ensures that they work together
according to the sweep configuration you defined.
{{% /alert %}}

The proceeding cell activates a sweep agent that runs the training function (`train`) 5 times:


```python
wandb.agent(sweep_id, train, count=5)
```

{{% alert %}}
Since the `random` search method was specified in the sweep configuration, the sweep controller provides randomly generated hyperparameter values.
{{% /alert %}}

For more information on how to create W&B Sweeps in a terminal, see the [W&B Sweep walkthrough]({{< relref "/guides/models/sweeps/walkthrough" >}}).

## Visualize Sweep Results



### Parallel Coordinates Plot
This plot maps hyperparameter values to model metrics. It’s useful for honing in on combinations of hyperparameters that led to the best model performance.

{{< img src="/images/tutorials/sweeps-2.png" alt="Sweep agent execution results" >}}


### Hyperparameter Importance Plot
The hyperparameter importance plot surfaces which hyperparameters were the best predictors of your metrics.
We report feature importance (from a random forest model) and correlation (implicitly a linear model).

{{< img src="/images/tutorials/sweeps-3.png" alt="W&B sweep dashboard" >}}

These visualizations can help you save both time and resources running expensive hyperparameter optimizations by honing in on the parameters (and value ranges) that are the most important, and thereby worthy of further exploration.


## Learn more about W&B Sweeps

We created a simple training script and [a few flavors of sweep configs](https://github.com/wandb/examples/tree/master/examples/keras/keras-cnn-fashion) for you to play with. We highly encourage you to give these a try.

That repo also has examples to help you try more advanced sweep features like [Bayesian Hyperband](https://app.wandb.ai/wandb/examples-keras-cnn-fashion/sweeps/us0ifmrf?workspace=user-lavanyashukla), and [Hyperopt](https://app.wandb.ai/wandb/examples-keras-cnn-fashion/sweeps/xbs2wm5e?workspace=user-lavanyashukla).