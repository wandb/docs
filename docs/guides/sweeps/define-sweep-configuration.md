---
description: Learn how to create configuration files for sweeps.
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Define sweep configuration

<head>
  <title>Define sweep configuration for hyperparameter tuning.</title>
</head>

A W&B Sweep combines a strategy for exploring hyperparameter values with the code that evaluates them. The strategy can be as simple as trying every option or as complex as Bayesian Optimization and Hyperband ([BOHB](https://arxiv.org/abs/1807.01774)).


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

## Sweep configuration structure

Define your sweep configuration with either a Python dictionary or a YAML file. Both sweep configuration format options utilize key-value pairs and nested structures. The top-level key of the Python dictionary or YAML is used to specify the [INSERT]. While values you specify in the (key-value pair) constrain the search space. [INSERT] 

:::info
See the respective documentation for more information on syntax. [LINK]
:::

:::tip
Use a YAML file if you want to create sweeps from the command line (CLI). Use Python dictionaries if you use a Jupyter Notebook or Python script. 
:::

For example, use top-level keys within your sweep configuration to define qualities of your sweep search such as the name of the sweep (`name` key), the parameters to search through (`parameter` key), the methodology to search the parameter space (`method` key), and more. The values you pass to the sweep configuration keys can include a dictionary of minimum and maximum learning rate values (in the case of a Python dictionary config) or a list of batch size values to search over (in the case of a YAML config).


The proceeding code snippets show example sweep configurations:

<Tabs
  defaultValue="yaml"
  values={[    
    {label: 'YAML', value: 'yaml'},
    {label: 'Python script or Jupyter Notebook', value: 'script'},
  ]}>
  <TabItem value="script">
  Define a sweep in a Python dictionary data structure if your training algorithm is defined in a Python script or Colab notebook.

```python title="train.py"
sweep_configuration = {
    "method": "bayes",
    "name": "sweepdemo",
    "metric": {
      "goal": "minimize",
      "name": "validation_loss"
      },
    "parameters": {
      "learning_rate": {"min": 0.0001, "max": 0.1},
      "batch_size": {"values": [16, 32, 64]},
      "epochs": {"values": [5, 10, 15]},
      "optimizer": {"values": ["adam", "sgd"]}
    },
}
```
  </TabItem>
  <TabItem value="yaml">


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


## Define a sweep configuration
1. First, provide the name of your training script and the search method for the program and method keys, respectively. 
  ```yaml title="sweep-config.yaml"
  program: train.py
  method: bayes
  ```
2. (Recommended) Specify the the metric to optimize (only used by certain search strategies and stopping criteria).
  ```yaml title="sweep-config.yaml"
  program: train.py
  method: bayes
  metric:
    goal: minimize
    name: validation_loss
  ```
2. Next, provide the name of your hyperparameter as a top-level key. Common hyperparameters include: learning rate, batch size, and epochs.
  ```yaml title="sweep-config.yaml"
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



## Top-level sweep configuration keys

The top-level sweep configuration keys are listed and briefly described below. See the respective sections for more information about each key. [LINK]

| Key               | Description                                                                                                                   |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `program`         | (required) Training script to run.                                                                                            |
| `entity`          | Specify the entity for this sweep.                                                                                            |
| `project`         | Specify the project for this sweep.                                                                                           |
| `description`     | Text description of the sweep.                                                                                                |
| `name`            | The name of the sweep, displayed in the W&B UI.                                                                               |
| [`method`](#method) | (required) Specify the [search strategy](./define-sweep-configuration.md#configuration-keys).                                 |
| [`metric`](#metric) | Specify the metric to optimize (only used by certain search strategies and stopping criteria).                                |
| [`parameters`](#parameters) | (required) Specify [parameters](define-sweep-configuration.md#parameters) bounds to search.                                   |
| [`early_terminate`](#earlyterminate) | Specify any [early stopping criteria](./define-sweep-configuration.md#early_terminate).                                       |
| [`command`](#command)         | Specify [command structure ](./define-sweep-configuration.md#command)for invoking and passing arguments to the training script. |
| `run_cap` | Specify a maximum number of runs in a sweep.                                                                                          |


The following sections describe in greater detail how to specify [INSERT].

### `method`

Specify the hyperparameter search strategy with the `method` key in your sweep configuration.

The following table describes available hyperparameter search methods:


| `method` | Description                                                                                           |
| -------- | ----------------------------------------------------------------------------------------------------- |
| `grid`   | Iterate over every combination of hyperparameter values. Can be computationally costly.               |
| `random` | Choose a random set of hyperparameter values on each iteration based on provided distributions.       |
| `bayes`  | Create a probabilistic model of a metric score as a function of the hyperparameters, and choose parameters with high probability of improving the metric. Bayesian hyperparameter search method uses a Gaussian Process to model the relationship between the parameters and the model metric and chooses parameters to optimize the probability of improvement. This strategy requires the `metric`key to be specified. Works well for small numbers of continuous parameters but scales poorly. |

:::caution
Random and Bayesian searches will run forever unless you stop the process from the command line, within your python script, or [the W&B App UI](./sweeps-ui.md). Grid search will also run forever if it searches within in a continuous search space.
:::

:::info
See the next section if you chose a random or Bayesian search method.
:::

#### Specify a distribution for random or Bayesian search method

Specify how to distribute values if you choose a random (`random`) or Bayesian (`bayes`) search method.

| Value                    | Description            |
| ------------------------ | ------------------------------------ |
| `constant`               | Constant distribution. Must specify `value`.                         |
| `categorical`            | Categorical distribution. Must specify `values`.                     |
| `int_uniform`            | Discrete uniform distribution on integers. Must specify `max` and `min` as integers.     |
| `uniform`                | Continuous uniform distribution. Must specify `max` and `min` as floats.      |
| `q_uniform`              | Quantized uniform distribution. Returns `round(X / q) * q` where X is uniform. `q` defaults to `1`.|
| `log_uniform`            | Log-uniform distribution. Returns a value `X` between `exp(min)` and `exp(max)`such that the natural logarithm is uniformly distributed between `min` and `max`.   |
| `log_uniform_values`     | Log-uniform distribution. Returns a value `X` between `min` and `max` such that `log(`X`)` is uniformly distributed between `log(min)` and `log(max)`.     |
| `q_log_uniform`          | Quantized log uniform. Returns `round(X / q) * q` where `X` is `log_uniform`. `q` defaults to `1`.       |
| `q_log_uniform_values`   | Quantized log uniform. Returns `round(X / q) * q` where `X` is `log_uniform_values`. `q` defaults to `1`.     |
| `inv_log_uniform`        | Inverse log uniform distribution. Returns `X`, where  `log(1/X)` is uniformly distributed between `min` and `max`.           |
| `inv_log_uniform_values` | Inverse log uniform distribution. Returns `X`, where  `log(1/X)` is uniformly distributed between `log(1/max)` and `log(1/min)`.    |
| `normal`                 | Normal distribution. Return value is normally-distributed with mean `mu` (default `0`) and standard deviation `sigma` (default `1`).|
| `q_normal`               | Quantized normal distribution. Returns `round(X / q) * q` where `X` is `normal`. Q defaults to 1.      |
| `log_normal`             | Log normal distribution. Returns a value `X` such that the natural logarithm `log(X)` is normally distributed with mean `mu` (default `0`) and standard deviation `sigma` (default `1`). |
| `q_log_normal`  | Quantized log normal distribution. Returns `round(X / q) * q` where `X` is `log_normal`. `q` defaults to `1`.             |
#### Examples

<Tabs
  defaultValue="constant"
  values={[
    {label: 'constant', value: 'constant'},
    {label: 'categorical', value: 'categorical'},
    {label: 'uniform', value: 'uniform'},
    {label: 'q_uniform', value: 'q_uniform'}
  ]}>
  <TabItem value="constant">

```yaml
parameter_name:
  distribution: constant
  value: 2.71828
```
  </TabItem>
  <TabItem value="categorical">

```yaml
parameter_name:
  distribution: categorical
  values:
      - elu
      - relu
      - gelu
      - selu
      - relu
      - prelu
      - lrelu
      - rrelu
      - relu6
```
  </TabItem>
  <TabItem value="uniform">

```yaml
parameter_name:
  distribution: uniform
  min: 0
  max: 1
```
  </TabItem>
  <TabItem value="q_uniform">

```yaml
parameter_name:
  distribution: q_uniform
  min: 0
  max: 256
  q: 1
```
  </TabItem>
</Tabs>


### `metric`

Use the metric top-level sweep configuration key to describe how to optimize the metric.

|Key | Description |
| -------- | --------------------------------------------------------- |
| `name`   | Name of the metric to optimize.                           |
| `goal`   | Either `minimize` or `maximize` (Default is `minimize`).  |
| `target` | Goal value for the metric you're optimizing. When any run in the sweep achieves that target value, the sweep's state will be set to `finished`. This means all agents with active runs will finish those jobs, but no new runs will be launched in the sweep. |

:::info
Ensure to log the metric you specify in you sweep configuration explicitly to W&B by your training script. For example, if you want to minimize the validation loss of your model:

```python
import wandb

run = wandb.init(entity="<entity>", project="<project>")

# Training script goes here

# model training code that returns validation loss as valid_loss
run.log({"val_loss": valid_loss})
```

<!-- #### Examples

<Tabs
  defaultValue="maximize"
  values={[
    {label: 'Maximize', value: 'maximize'},
    {label: 'Minimize', value: 'minimize'},
    {label: 'Target', value: 'target'},
  ]}>
  <TabItem value="maximize">

```yaml
metric:
  name: val_acc
  goal: maximize
```
  </TabItem>
  <TabItem value="minimize">

```yaml
metric:
  name: val_loss
  goal: minimize
```
  </TabItem>
  <TabItem value="target">

```yaml
metric:
  name: val_acc
  goal: maximize
  target: 0.95
```
  </TabItem>
</Tabs> -->

Do not log the metric for your sweep inside of a sub-directory. In the proceeding code example, a user wants to log the validation loss (`"loss": val_loss`). First they pass the values into a dictionary. However, the dictionary passed to `wandb.log` does not specify the key-value pair to track.

```python
val_metrics = {
        "loss": val_loss, 
        "acc": val_acc
        }

# Incorrect. Dictionary key-value paired is not provided.
wandb.log({"val_loss", val_metrics})
```

Instead, log the metric at the top level. For example, after you create a dictionary, specify the key-value pair when you pass the dictionary to the `wandb.log` method:

```python
val_metrics = {
        "loss": val_loss, 
        "acc": val_acc
        }

wandb.log({"val_loss", val_metrics["loss"]})
```

:::





### `parameters`

Specify one or more hyperparameters to explore during a sweep. 

For each hyperparameter, specify the name and the possible values as a list of constants (for any `method`) or specify a `distribution` for `random` or `bayes`.

| Values          | Description                                                             |
| --------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `values`        | Specifies all valid values for this hyperparameter. Compatible with `grid`.     |
| `value`         | Specifies the single valid value for this hyperparameter. Compatible with `grid`.                                                                                               |
| `distribution`  | (`str`) Selects a distribution from the distribution table below. If not specified, will default to `categorical` if `values` is set, to `int_uniform` if `max` and `min` are set to integers, to `uniform` if `max` and `min` are set to floats, or to`constant` if `value` is set. See the [INSERT] for more information. |
| `probabilities` | Specify the probability of selecting each element of `values` when using `random`.                                |
| `min`, `max`    | (`int`or `float`) Maximum and minimum values. If `int`, for `int_uniform` -distributed hyperparameters. If `float`, for `uniform` -distributed hyperparameters.                |
| `mu`            | (`float`) Mean parameter for `normal` - or `lognormal` -distributed hyperparameters.                                                       |
| `sigma`         | (`float`) Standard deviation parameter for `normal` - or `lognormal` -distributed hyperparameters.                            |
| `q`             | (`float`) Quantization step size for quantized hyperparameters.                          |
| `parameters`    | Nest other parameters inside a root level parameter.           |
<!-- 
#### Examples

<Tabs
  defaultValue="single"
  values={[
    {label: 'single value', value: 'single'},
    {label: 'multiple values', value: 'multiple'},
    {label: 'probabilities', value: 'probabilities'},
    {label: 'distribution', value: 'distribution'},
    {label: 'nested', value: 'nested'},
  ]}>
  <TabItem value="single">

  ```yaml
  parameter_name:
    value: 1.618
  ```

  </TabItem>
  <TabItem value="multiple">

  ```yaml
  parameter_name:
    values:
      - 8
      - 6
      - 7
      - 5
      - 3
      - 0
      - 9
  ```
  </TabItem>
  <TabItem value="probabilities">

  ```yaml
  parameter_name:
    values: [1, 2, 3, 4, 5]
    probabilities: [0.1, 0.2, 0.1, 0.25, 0.35]
  ```

  </TabItem>
  <TabItem value="distribution">

  ```yaml
  parameter_name:
    distribution: normal
    mu: 100
    sigma: 10
  ```

  </TabItem>
  <TabItem value="nested">

  ```yaml
  top_level_param:
    min: 0
    max: 5
  nested_param:
      parameters:  # required key
          learning_rate:
              values: [0.01, 0.001]
          double_nested_param:
            parameters:  # <--
              x:
                value: 0.9
              y: 
                value: 0.8
  ```

  </TabItem>
</Tabs> -->

### `early_terminate`

Early termination is an optional feature that speeds up hyperparameter search by stopping poorly-performing runs. When the early stopping is triggered, the agent stops the current run and gets the next set of hyperparameters to try.

| Key    | Description                    |
| ------ | ------------------------------ |
| `type` | Specify the stopping algorithm |

We support the following stopping algorithm(s):

| `type`      | Description                                                   |
| ----------- | ------------------------------------------------------------- |
| `hyperband` | Use the [hyperband method](https://arxiv.org/abs/1603.06560). |

#### Use `hyperband` to stop a hyperparameter search

[Hyperband](https://arxiv.org/abs/1603.06560) stopping evaluates if a program should be stopped or permitted to continue at one or more pre-set iteration counts, called "brackets". When a run reaches a bracket, its metric value is compared to all previous reported metric values and the [W&B Run](../../ref/python/run.md) is terminated if its value is too high (when the goal is minimization) or low (when the goal is maximization).

Brackets are based on the number of logged iterations. The number of brackets corresponds to the number of times you log the metric you are optimizing. The iterations can correspond to steps, epochs, or something in between. The numerical value of the step counter is not used in bracket calculations.

:::caution
Specify either `min_iter` or `max_iter` to create a bracket schedule.
:::


| Key        | Description                                                    |
| ---------- | -------------------------------------------------------------- |
| `min_iter` | Specify the iteration for the first bracket                    |
| `max_iter` | Specify the maximum number of iterations.                      |
| `s`        | Specify the total number of brackets (required for `max_iter`) |
| `eta`      | Specify the bracket multiplier schedule (default: `3`).        |
| `strict`   | Enable 'strict' mode that prunes runs aggressively, more closely following the original Hyperband paper. Defaults to false. |

:::info
The hyperband early terminator checks what [W&B Runs](../../ref/python/run.md) to terminate once every few minutes. The end run timestamp might differ from the specified brackets if your run or iteration are short.
:::


#### Examples

<Tabs
  defaultValue="min_iter"
  values={[
    {label: 'Hyperband (min_iter)', value: 'min_iter'},
    {label: 'Hyperband (max_iter)', value: 'max_iter'},
  ]}>
  <TabItem value="min_iter">

```yaml
early_terminate:
  type: hyperband
  min_iter: 3
```

The brackets for this example are: `[3, 3*eta, 3*eta*eta, 3*eta*eta*eta]`, which equals `[3, 9, 27, 81]`.
  </TabItem>
  <TabItem value="max_iter">

```yaml
early_terminate:
  type: hyperband
  max_iter: 27
  s: 2
```

The brackets for this example are `[27/eta, 27/eta/eta]`, which equals `[9, 3]`.
  </TabItem>
</Tabs>

### `command` 

<!-- Agents created with [`wandb agent`](../../ref/cli/wandb-agent.md) receive a command in the following format by default: -->

<Tabs
  defaultValue="unix"
  values={[
    {label: 'UNIX', value: 'unix'},
    {label: 'Windows', value: 'windows'},
  ]}>
  <TabItem value="unix">

```bash
/usr/bin/env python train.py --param1=value1 --param2=value2
```
  </TabItem>
  <TabItem value="windows">

```bash
python train.py --param1=value1 --param2=value2
```
  </TabItem>
</Tabs>

:::info
On UNIX systems, `/usr/bin/env` ensures the right Python interpreter is chosen based on the environment.
:::

The format and contents can be modified by specifying values under the `command` key. Fixed components of the command, such as filenames, can be included directly (see examples below).

We support the following macros for variable components of the command:

| Command Macro              | Description                                                                                                                                                           |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `${env}`                   | `/usr/bin/env` on UNIX systems, omitted on Windows.                                                                                                                   |
| `${interpreter}`           | Expands to `python`.                                                                                                                                                  |
| `${program}`               | Training script filename specified by the sweep configuration `program` key.                                                                                          |
| `${args}`                  | Hyperparameters and their values in the form `--param1=value1 --param2=value2`.                                                                                       |
| `${args_no_boolean_flags}` | Hyperparameters and their values in the form `--param1=value1` except boolean parameters are in the form `--boolean_flag_param` when `True` and omitted when `False`. |
| `${args_no_hyphens}`       | Hyperparameters and their values in the form `param1=value1 param2=value2`.                                                                                           |
| `${args_json}`             | Hyperparameters and their values encoded as JSON.                                                                                                                     |
| `${args_json_file}`        | The path to a file containing the hyperparameters and their values encoded as JSON.                                                                                   |
| `${envvar}`                | A way to pass environment variables. `${envvar:MYENVVAR}` __ expands to the value of MYENVVAR environment variable. __                                               |

The default command format is defined as:

```yaml
command:
  - ${env}
  - ${interpreter}
  - ${program}
  - ${args}
```

#### Examples

<Tabs
  defaultValue="python"
  values={[
    {label: 'Set python interpreter', value: 'python'},
    {label: 'Add extra parameters', value: 'parameters'},
    {label: 'Omit arguments', value: 'omit'},
    {label: 'Hydra', value: 'hydra'}
  ]}>
  <TabItem value="python">

Remove the `{$interpreter}` macro and provide a value explicitly in order to hardcode the python interpreter. For example, the following code snippet demonstrates how to do this:

```yaml
command:
  - ${env}
  - python3
  - ${program}
  - ${args}
```
  </TabItem>
  <TabItem value="parameters">

To add extra command line arguments not specified by sweep configuration parameters:

```
command:
  - ${env}
  - ${interpreter}
  - ${program}
  - "--config"
  - "your-training-config.json"
  - ${args}
```

  </TabItem>
  <TabItem value="omit">

If your program does not use argument parsing you can avoid passing arguments all together and take advantage of `wandb.init` picking up sweep parameters into `wandb.config` automatically:

```
command:
  - ${env}
  - ${interpreter}
  - ${program}
```
  </TabItem>
  <TabItem value="hydra">

You can change the command to pass arguments the way tools like [Hydra](https://hydra.cc) expect. See [Hydra with W&B](../integrations/other/hydra.md) for more information.

```
command:
  - ${env}
  - ${interpreter}
  - ${program}
  - ${args_no_hyphens}
```
  </TabItem>
</Tabs>

## Nested Parameters

The sweep configuration format supports specifying nested parameters. To delineate a nested parameter, use an additional `parameters` key under the top level parameter name. Multi-level nesting is allowed. For a complete sweep configuration example: 

```yaml
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
      parameters:  # <--
        x:
          value: 0.9
        y: 
          value: 0.8
```

The above configuration might result in the following run config (in `train.py`):

```json
{
  "top_level_param": 0,
  "nested_param": {
    "learning_rate": 0.01,
    "double_nested_param": {
      "x": 0.9,
      "y": 0.8
    }
  }
}
```

:::caution
Nested parameters overwrite keys specified in a run configuration.

For example, suppose you initialize a W&B run with the following configuration:

```python
run = wandb.init(config={"nested_param": {"manual_key": 1}})
```

If you then create a sweep with the following sweep configuration:

```json
{
  "top_level_param": 0,
  "nested_param": {
    "learning_rate": 0.01,
    "double_nested_param": {
      "x": 0.9,
      "y": 0.8
    }
  }
}
```

The `nested_param.manual_key` that was passed when the W&B run was initialized will not be available. Instead, the `run.config` will look exactly like the JSON code snippet shown above.
:::

