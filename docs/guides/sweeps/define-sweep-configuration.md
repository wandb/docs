---
description: Learn how to create configuration files for sweeps.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Sweep configuration structure

<head>
  <title>Define sweep configuration for hyperparameter tuning.</title>
</head>

A W&B Sweep combines a strategy for exploring hyperparameter values with the code that evaluates them. The strategy can be as simple as trying every option or as complex as Bayesian Optimization and Hyperband ([BOHB](https://arxiv.org/abs/1807.01774)).

The following guide describes how to format your sweep configuration. See [Sweep configuration options](./sweep-config-keys.md) for a comprehensive list of top-level sweep configuration keys.

## Basic structure

Define your sweep configuration with either a [Python dictionary](https://docs.python.org/3/tutorial/datastructures.html#dictionaries) or a [YAML](https://yaml.org/) file. Use a YAML file if you want to manage sweeps interactively from the command line (CLI). Use Python dictionaries if you use a Jupyter Notebook or Python script. Both sweep configuration format options utilize key-value pairs and nested structures. 

Use top-level keys within your sweep configuration to define qualities of your sweep search such as the name of the sweep ([`name`](./sweep-config-keys.md#name) key), the parameters to search through ([`parameter`](./sweep-config-keys.md#parameters) key), the methodology to search the parameter space ([`method`](./sweep-config-keys.md#method) key), and more. 


The proceeding code snippets show the same sweep configuration defined within a YAML file and within a Python script or Jupyter notebook.

<Tabs
  defaultValue="cli"
  values={[    
    {label: 'CLI', value: 'cli'},
    {label: 'Python script or Jupyter notebook', value: 'script'},
  ]}>
  <TabItem value="script">

Define a sweep in a Python dictionary data structure if your training algorithm is defined in a Python script or Jupyter notebook. 

The proceeding code snippet stores a sweep configuration in a variable named `sweep_configuration`:

```python title="train.py"
sweep_configuration = {
    "name": "sweepdemo",
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "validation_loss"},
    "parameters": {
        "learning_rate": {"min": 0.0001, "max": 0.1},
        "batch_size": {"values": [16, 32, 64]},
        "epochs": {"values": [5, 10, 15]},
        "optimizer": {"values": ["adam", "sgd"]},
    },
}
```
  </TabItem>
  <TabItem value="cli">
Define a sweep configuration in a YAML file if you want to manage sweeps interactively from the command line (CLI)

```yaml title="config.yaml"
program: train.py
name: sweepdemo
method: bayes
metric:
  goal: minimize
  name: validation_loss
parameters:
  learning_rate:
    min: 0.0001
    max: 0.1
  batch_size:
    values: [16, 32, 64]
  epochs:
    values: [5, 10, 15]
  optimizer:
    values: ["adam", "sgd"]
```
  </TabItem>
</Tabs>

For more information on available top level keys, see [Sweep configuration options](./sweep-config-keys.md).

## Nested Parameters

Sweep configurations support nested parameters. To delineate a nested parameter, use an additional `parameters` key under the top level parameter name. Multi-level nesting is allowed. 

The proceeding code snippets show how to define a sweep configuration with a YAML file and within a Python script or Jupyter notebook. 

<Tabs
  defaultValue="cli"
  values={[
    {label: 'CLI', value: 'cli'},
    {label: 'Python script or Jupyter notebook', value: 'notebook'},
  ]}>
  <TabItem value="cli">

```yaml title="config.yaml"
program: train.py
method: bayes
metric:
  name: val_loss
  goal: minimize
top_level_param:
  min: 0
  max: 5
nested_param:
  parameters:  # required key
    learning_rate:
      values: [0.01, 0.001]
    double_nested_param:
      parameters: 
        x:
          value: 0.9
        y: 
          value: 0.8
```

  </TabItem>
  <TabItem value="notebook">

The proceeding code snippet stores a sweep configuration, that has nested parameters, in a variable named `sweep_configuration`:

```python
sweep_configuration = {
    "top_level_param": {"min": 0, "max": 5},
    "nested_param": {
        "learning_rate": 0.01,
        "double_nested_param": {"x": 0.9, "y": 0.8},
    },
}
```

  </TabItem>
</Tabs>


:::caution
Nested parameters defined in sweep configuration overwrite keys specified in a W&B run configuration.

For example, suppose you initialize a W&B run with the following configuration in a `train.py` Python script (see Lines 1-2). Next, you define a sweep configuration in a dictionary called `sweep_configuration` (see Lines 4-13). You then pass the sweep config dictionary to `wandb.sweep` to initialize a sweep config (see Line 16).


```python title="train.py" showLineNumbers
def main():
    run = wandb.init(config={"nested_param": {"manual_key": 1}})


sweep_configuration = {
    "top_level_param": 0,
    "nested_param": {
        "learning_rate": 0.01,
        "double_nested_param": {"x": 0.9, "y": 0.8},
    },
}

# Initialize sweep by passing in config.
sweep_id = wandb.sweep(sweep=sweep_configuration, project="<project>")

# Start sweep job.
wandb.agent(sweep_id, function=main, count=4)
```
The `nested_param.manual_key` that was passed when the W&B run was initialized (line 2) will not be available. Instead, the `run.config` will only possess the key-value pairs that were defined in the sweep configuration dictionary (lines 4-13).

:::


## Consideration when logging metrics 



Ensure to log the metric you specify in your sweep configuration explicitly to W&B. Do not log metrics for your sweep inside of a sub-directory. 

For example, consider the proceeding psuedocode. A user wants to log the validation loss (`"val_loss": loss`). First they pass the values into a dictionary (line 16). However, the dictionary passed to `wandb.log` does not explicitly access the key-value pair in the dictionary:

```python title="train.py" showLineNumbers
# Import the W&B Python Library and log into W&B
import wandb
import random

def train():
    offset = random.random() / 5
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset        

    val_metrics = {"val_loss": loss, "val_acc": acc}
    return val_metrics

def main():
    wandb.init(entity="<entity>", project="my-first-sweep")
    val_metrics = train()
    # highlight-next-line
    wandb.log({"val_loss": val_metrics})

sweep_configuration = {
    "method": "random",
    "metric": {
        "goal": "minimize",
        "name": "val_loss"
        },
    "parameters": {
        "x": {"max": 0.1, "min": 0.01},
        "y": {"values": [1, 3, 7]},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")

wandb.agent(sweep_id, function=main, count=10)
```

Instead, explicitly access the key-value pair within the Python dictionary. For example, the proceeding code (line after you create a dictionary, specify the key-value pair when you pass the dictionary to the `wandb.log` method:

```python title="train.py" showLineNumbers
# Import the W&B Python Library and log into W&B
import wandb
import random

def train():
    offset = random.random() / 5
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset        

    val_metrics = {"val_loss": loss, "val_acc": acc}
    return val_metrics

def main():
    wandb.init(entity="<entity>", project="my-first-sweep")
    val_metrics = train()
    # highlight-next-line
    wandb.log({"val_loss", val_metrics["val_loss"]})

sweep_configuration = {
    "method": "random",
    "metric": {
        "goal": "minimize",
        "name": "val_loss"
        },
    "parameters": {
        "x": {"max": 0.1, "min": 0.01},
        "y": {"values": [1, 3, 7]},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")

wandb.agent(sweep_id, function=main, count=10)
```




<!-- :::caution
1. Ensure that you log (`wandb.log`) the _exact_ metric name that you define in your sweep configuration.
2. You cannot change the Sweep configuration once you start the W&B Sweep agent.
:::

For example, suppose you want W&B Sweeps to maximize the validation accuracy during training. Within your Python script you store the validation accuracy in a variable `val_loss`. In your YAML configuration file you define this as:

```yaml
metric:
  goal: maximize
  name: val_loss
```

You must log the variable `val_loss` (in this example) within your Python script or Jupyter Notebook to W&B.

```python
wandb.log({"val_loss": validation_loss})
```

Defining the metric in the sweep configuration is only required if you use the bayes method for the sweep.  -->