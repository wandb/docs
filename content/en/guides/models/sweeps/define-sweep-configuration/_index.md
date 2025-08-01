---
description: Learn how to create configuration files for sweeps.
menu:
  default:
    identifier: define-sweep-configuration
    parent: sweeps
url: guides/sweeps/define-sweep-configuration
title: Define a sweep configuration
weight: 3
---


A W&B Sweep combines a strategy for exploring hyperparameter values with the code that evaluates them. The strategy can be as simple as trying every option or as complex as Bayesian Optimization and Hyperband ([BOHB](https://arxiv.org/abs/1807.01774)).

Define a sweep configuration either in a [Python dictionary](https://docs.python.org/3/tutorial/datastructures.html#dictionaries) or a [YAML](https://yaml.org/) file. How you define your sweep configuration depends on how you want to manage your sweep.

{{% alert %}}
Define your sweep configuration in a YAML file if you want to initialize a sweep and start a sweep agent from the command line. Define your sweep in a Python dictionary if you initialize a sweep and start a sweep entirely within a Python script or notebook.
{{% /alert %}}

The following guide describes how to format your sweep configuration. See [Sweep configuration options]({{< relref "./sweep-config-keys.md" >}}) for a comprehensive list of top-level sweep configuration keys.

## Basic structure

Both sweep configuration format options (YAML and Python dictionary) utilize key-value pairs and nested structures. 

Use top-level keys within your sweep configuration to define qualities of your sweep search such as the name of the sweep ([`name`]({{< relref "./sweep-config-keys.md" >}}) key), the parameters to search through ([`parameters`]({{< relref "./sweep-config-keys.md#parameters" >}}) key), the methodology to search the parameter space ([`method`]({{< relref "./sweep-config-keys.md#method" >}}) key), and more. 


For example, the proceeding code snippets show the same sweep configuration defined within a YAML file and within a Python dictionary. Within the sweep configuration there are five top level keys specified: `program`, `name`, `method`, `metric` and `parameters`. 


{{< tabpane  text=true >}}
  {{% tab header="CLI" %}}
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
  {{% /tab %}}
  {{% tab header="Python script or notebook" %}}
Define a sweep in a Python dictionary data structure if you define training algorithm in a Python script or notebook. 

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
  {{% /tab %}}
{{< /tabpane >}}


Within the top level `parameters` key, the following keys are nested: `learning_rate`, `batch_size`, `epoch`, and `optimizer`. For each of the nested keys you specify, you can provide one or more values, a distribution, a probability, and more. For more information, see the [parameters]({{< relref "./sweep-config-keys.md#parameters" >}}) section in [Sweep configuration options]({{< relref "./sweep-config-keys.md" >}}). 


## Double nested parameters

Sweep configurations support nested parameters. To delineate a nested parameter, use an additional `parameters` key under the top level parameter name. Sweep configs support multi-level nesting.

Specify a probability distribution for your random variables if you use a Bayesian or random hyperparameter search. For each hyperparameter:

1. Create a top level `parameters` key in your sweep config.
2. Within the `parameters`key, nest the following:
   1. Specify the name of hyperparameter you want to optimize. 
   2. Specify the distribution you want to use for the `distribution` key. Nest the `distribution` key-value pair underneath the hyperparameter name.
   3. Specify one or more values to explore. The value (or values) should be inline with the distribution key.  
      1. (Optional) Use an additional parameters key under the top level parameter name to delineate a nested parameter.

<!-- For example, the proceeding code snippets show a sweep config both in a YAML config file and a Python script.   -->


<!-- To do: what is a double-nested parameter -->



{{% alert color="secondary" %}}
Nested parameters defined in sweep configuration overwrite keys specified in a W&B run configuration.

For example, suppose you initialize a W&B run with the following configuration in a `train.py` Python script (see Lines 1-2). Next, you define a sweep configuration in a dictionary called `sweep_configuration` (see Lines 4-13). You then pass the sweep config dictionary to `wandb.sweep` to initialize a sweep config (see Line 16).


```python title="train.py" 
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
The `nested_param.manual_key` that is passed when the W&B run is initialized is not accessible. The `wandb.Run.config` only possess the key-value pairs that are defined in the sweep configuration dictionary.
{{% /alert %}}


## Sweep configuration template


The following template shows how you can configure parameters and specify search constraints. Replace `hyperparameter_name` with the name of your hyperparameter and any values enclosed in `<>`.

```yaml title="config.yaml"
program: <insert>
method: <insert>
parameter:
  hyperparameter_name0:
    value: 0  
  hyperparameter_name1: 
    values: [0, 0, 0]
  hyperparameter_name: 
    distribution: <insert>
    value: <insert>
  hyperparameter_name2:  
    distribution: <insert>
    min: <insert>
    max: <insert>
    q: <insert>
  hyperparameter_name3: 
    distribution: <insert>
    values:
      - <list_of_values>
      - <list_of_values>
      - <list_of_values>
early_terminate:
  type: hyperband
  s: 0
  eta: 0
  max_iter: 0
command:
- ${Command macro}
- ${Command macro}
- ${Command macro}
- ${Command macro}      
```

To express a numeric value using scientific notation, add the YAML `!!float` operator, which casts the value to a floating point number. For example, `min: !!float 1e-5`. See [Command example]({{< relref "#command-example" >}}).

## Sweep configuration examples

{{< tabpane text=true >}}
  {{% tab header="CLI" %}}

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

  {{% /tab %}}
  {{% tab header="Python script or notebook" %}}

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

  {{% /tab %}}
{{< /tabpane >}}



### Bayes hyperband example

```yaml
program: train.py
method: bayes
metric:
  goal: minimize
  name: val_loss
parameters:
  dropout:
    values: [0.15, 0.2, 0.25, 0.3, 0.4]
  hidden_layer_size:
    values: [96, 128, 148]
  layer_1_size:
    values: [10, 12, 14, 16, 18, 20]
  layer_2_size:
    values: [24, 28, 32, 36, 40, 44]
  learn_rate:
    values: [0.001, 0.01, 0.003]
  decay:
    values: [1e-5, 1e-6, 1e-7]
  momentum:
    values: [0.8, 0.9, 0.95]
  epochs:
    value: 27
early_terminate:
  type: hyperband
  s: 2
  eta: 3
  max_iter: 27
```

The proceeding tabs show how to specify either a minimum or maximum number of iterations for `early_terminate`:

{{< tabpane  text=true >}}
  {{% tab header="Maximum number of iterations" %}}

The brackets for this example are: `[3, 3*eta, 3*eta*eta, 3*eta*eta*eta]`, which equals `[3, 9, 27, 81]`.  

```yaml
early_terminate:
  type: hyperband
  min_iter: 3
```

  {{% /tab %}}
  {{% tab header="Minimum number of iterations" %}}

The brackets for this example are `[27/eta, 27/eta/eta]`, which equals `[9, 3]`. 

```yaml
early_terminate:
  type: hyperband
  max_iter: 27
  s: 2
```

  {{% /tab %}}
{{< /tabpane >}}



### Macro and custom command arguments example

For more complex command line arguments, you can use macros to pass environment variables, the Python interpreter, and additional arguments. [W&B supports pre defined macros]({{< relref "./sweep-config-keys.md#command-macros" >}}) and custom command line arguments that you can specify in your sweep configuration.

For example, the following sweep configuration (`sweep.yaml`) defines a command that runs a Python script (`run.py`) with the `${env}`, `${interpreter}`, and `${program}` macros replaced with the appropriate values when the sweep runs.

The `--batch_size=${batch_size}`, `--test=True`, and `--optimizer=${optimizer}` arguments use custom macros to pass the values of the `batch_size`, `test`, and `optimizer` parameters defined in the sweep configuration.

```yaml title="sweep.yaml"
program: run.py
method: random
metric:
  name: validation_loss
parameters:
  learning_rate:
    min: 0.0001
    max: 0.1
command:
  - ${env}
  - ${interpreter}
  - ${program}
  - "--batch_size=${batch_size}"
  - "--optimizer=${optimizer}"
  - "--test=True"
```
The associated Python script (`run.py`) can then parse these command line arguments using the `argparse` module. 

```python title="run.py"
# run.py  
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int)
parser.add_argument('--optimizer', type=str, choices=['adam', 'sgd'], required=True)
parser.add_argument('--test', type=str2bool, default=False)
args = parser.parse_args()

# Initialize a W&B Run
with wandb.init('test-project') as run:
    run.log({'validation_loss':1})
```

See the [Command macros]({{< relref "./sweep-config-keys.md#command-macros" >}}) section in [Sweep configuration options]({{< relref "./sweep-config-keys.md" >}}) for a list of pre-defined macros you can use in your sweep configuration. 

#### Boolean arguments

The `argparse` module does not support boolean arguments by default. To define a boolean argument, you can use the [`action`](https://docs.python.org/3/library/argparse.html#action) parameter or use a custom function to convert the string representation of the boolean value to a boolean type.

As an example, you can use the following code snippet to define a boolean argument. Pass `store_true` or `store_false` as an argument to `ArgumentParser`. 

```python
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

args.test  # This will be True if --test is passed, otherwise False
```

You can also define a custom function to convert the string representation of the boolean value to a boolean type. For example, the following code snippet defines the `str2bool` function, which converts a string to a boolean value. 

```python
def str2bool(v: str) -> bool:
  """Convert a string to a boolean. This is required because
  argparse does not support boolean arguments by default.
  """
  if isinstance(v, bool):
      return v
  return v.lower() in ('yes', 'true', 't', '1')
```





