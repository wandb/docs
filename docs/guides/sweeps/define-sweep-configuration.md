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


For example, the proceeding code snippets show the same sweep configuration defined within a YAML file and within a Python script or Jupyter notebook. Within the sweep configuration there are five top level keys specified: `program`, `name`, `method`, `metric` and `parameters`. 

Within the `parameters` top level key, the `learning_rate`, `batch_size`, `epoch`, and `optimizer` keys are nested. Values that are provided to the nested keys can take one or more `values`, a [`distribution`](./sweep-config-keys.md#distribution-options-for-random-and-bayesian-search), and more. 

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

## Double nested parameters

Sweep configurations support nested parameters. To delineate a nested parameter, use an additional `parameters` key under the top level parameter name. Multi-level nesting is allowed. 


Specify a probability distribution for your random variables if you use a Bayesian or random hyperparameter search. For each hyperparameter:

1. Create a top level `parameters` key in your sweep config.
2. Within the `parameters`key, nest the following:
   1. Specify the name of hyperparameter you want to optimize. 
   2. Specify the distribution you want to use for the `distribution` key. Nest the `distribution` key-value pair underneath the hyperparameter name.
   3. Specify one or more values to explore. The value (or values) should be inline with the distribution key.  
      1. (Optional) Use an additional parameters key under the top level parameter name to delineate a nested parameter.

For example, the proceeding code snippets show a sweep config both in a YAML config file and a Python script.  

[INSERT]

<!-- To do: what is a double-nested parameter -->

<Tabs
  defaultValue="cli"
  values={[
    {label: 'CLI', value: 'cli'},
    {label: 'Python script or Jupyter notebook', value: 'notebook'},
  ]}>
  <TabItem value="cli">


```yaml title="config.yaml" 
program: train.py
method: random
metric:
  goal: minimize
  name: loss
parameters:
  batch_size:
    distribution: q_log_uniform_values
    max: 256 
    min: 32
    q: 8
  dropout: 
    values: [0.3, 0.4, 0.5]
  epochs:
    value: 1
  fc_layer_size: 
    values: [128, 256, 512]
  learning_rate:
    distribution: uniform
    max: 0.1
    min: 0
  optimizer:
    values: ["adam", "sgd"]
```

  </TabItem>
  <TabItem value="notebook">

```python title="train.py" 
sweep_config = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "loss"},
    "parameters": {
        "batch_size": {
            "distribution": "q_log_uniform_values",
            "max": 256,
            "min": 32,
            "q": 8,
        },
        "dropout": {"values": [0.3, 0.4, 0.5]},
        "epochs": {"value": 1},
        "fc_layer_size": {"values": [128, 256, 512]},
        "learning_rate": {"distribution": "uniform", "max": 0.1, "min": 0},
        "optimizer": {"values": ["adam", "sgd"]},
    },
}
```

  </TabItem>

</Tabs>





<!-- The proceeding code snippets show how to define a sweep configuration with a YAML file and within a Python script or Jupyter notebook. 

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
</Tabs> -->


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



