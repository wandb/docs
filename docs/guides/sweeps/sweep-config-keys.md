---
sidebar_display: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Sweep configuration options

A sweep configuration is comprised of nested key-value pairs. Use top-level keys within your sweep configuration to define qualities of your sweep search such as the parameters to search through ([`parameter`](./sweep-config-keys.md#parameters) key), the methodology to search the parameter space ([`method`](./sweep-config-keys.md#method) key), and more. 

Top-level sweep configuration keys are listed and briefly described below. See the respective sections for more information about each key. 


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
| [`early_terminate`](#early_terminate) | Specify any [early stopping criteria](./define-sweep-configuration.md#early_terminate).                                       |
| [`command`](#command)         | Specify [command structure ](./define-sweep-configuration.md#command)for invoking and passing arguments to the training script. |
| `run_cap` | Specify a maximum number of runs in a sweep.                                                                                          |

See the [Sweep configuration](./sweep-config-keys.md) structure for more information on how to structure your sweep configuration.

## `program`

##  `entity`

## `project`

## `description`

## `name`


## `method`



Specify the hyperparameter search strategy with the `method` key. There are three hyperparameter search strategies to choose from: grid, random, and bayesian search. 


### Search method options


#### Grid search
Iterate over every combination of hyperparameter values.  Grid search makes uninformed decisions on the set of hyperparameter values to use on each iteration. Grid search can be computationally costly.     

Grid search will run forever if it searches within in a continuous search space.

#### Random search
Choose a random, uninformed, set of hyperparameter values on each iteration based on a distribution. Random search runs forever unless you stop the process from the command line, within your python script, or [the W&B App UI](./sweeps-ui.md).

Specify the distribution space with the metric key if you choose random (`method: random`) search.

#### Bayesian search
In contrast to [random](#random-search) and [grid](#grid-search) search, Bayesian models make informed decisions. Bayesian optimization uses a probabilistic model to decide which values to use through an iterative process of testing values on a surrogate function before evaluating the objective function. Bayesian search works well for small numbers of continuous parameters but scales poorly. For more information about Bayesian search, see [LINK]. 

<!-- There are different Bayesian optimization methods. W&B uses a Gaussian process to model the relationship between hyperparameters and the model metric. For more information, see this paper. [LINK] -->

Bayesian search runs forever unless you stop the process from the command line, within your python script, or [the W&B App UI](./sweeps-ui.md). 

Specify the distribution space with the metric key if you choose Bayesian (`method: bayes`) search method.

### Distribution options for random and Bayesian search

Specify a probability distribution for your random variables if you use a Bayesian or random hyperparameter search. For each hyperparameter:

1. Create a top level `parameters` key in your sweep config.
2. Within the `parameters`key, nest the following:
   1. Specify the name of hyperparameter you want to optimize. 
   2. Specify the distribution you want to use for the `distribution` key. Nest the `distribution` key-value pair underneath the hyperparameter name.
   3. Specify one or more values to explore. The value (or values) should be inline with the distribution key.

The general template to specify a distribution will look similar to the proceeding code snippet:

```python title="train.py"
sweep_config = {
    "method": "bayes",
    "metric" :  {
        "goal": "minimize", 
        "name": "validation_loss"
        },
    "parameters": {
        "learning_rate" : {
            "min": 0.0001
            "max": 0.01 
        },
        "batch_size": {
            "distribution": "q_log_uniform_values",
            "max": 256,
             "min": 32,
             "q": 8,
        }
    }
}
```

```yaml title="config.yaml"
program: train.py
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

The proceeding tables lists distributions W&B supports.
<!-- Nest the typekey within early_terminate within your sweep configuration.
For each hyperparameter, specify the name and the possible values as a list of constants (for any method) or specify a distribution for random or bayes. -->

[INSERT]

| Distribution             | Description            |
| ------------------------ | ------------------------------------ |
| `constant`               | Constant distribution. Must specify the constant value (`value`) to use.                    |
| `categorical`            | Categorical distribution. Must specify all valid values (`values`) for this hyperparameter. |
| `int_uniform`            | Discrete uniform distribution on integers. Must specify `max` and `min` as integers.     |
| `uniform`                | Continuous uniform distribution. Must specify `max` and `min` as floats.      |
| `q_uniform`              | Quantized uniform distribution. Returns `round(X / q) * q` where X is uniform. `q` defaults to `1`.|
| `log_uniform`            | Log-uniform distribution. Returns a value `X` between `exp(min)` and `exp(max)`such that the natural logarithm is uniformly distributed between `min` and `max`.   |
| `log_uniform_values`     | Log-uniform distribution. Returns a value `X` between `min` and `max` such that `log(`X`)` is uniformly distributed between `log(min)` and `log(max)`.     |
| `q_log_uniform`          | Quantized log uniform. Returns `round(X / q) * q` where `X` is `log_uniform`. `q` defaults to `1`. |
| `q_log_uniform_values`   | Quantized log uniform. Returns `round(X / q) * q` where `X` is `log_uniform_values`. `q` defaults to `1`.  |
| `inv_log_uniform`        | Inverse log uniform distribution. Returns `X`, where  `log(1/X)` is uniformly distributed between `min` and `max`. |
| `inv_log_uniform_values` | Inverse log uniform distribution. Returns `X`, where  `log(1/X)` is uniformly distributed between `log(1/max)` and `log(1/min)`.    |
| `normal`                 | Normal distribution. Return value is normally-distributed with mean `mu` (default `0`) and standard deviation `sigma` (default `1`).|
| `q_normal`               | Quantized normal distribution. Returns `round(X / q) * q` where `X` is `normal`. Q defaults to 1.  |
| `log_normal`             | Log normal distribution. Returns a value `X` such that the natural logarithm `log(X)` is normally distributed with mean `mu` (default `0`) and standard deviation `sigma` (default `1`). |
| `q_log_normal`  | Quantized log normal distribution. Returns `round(X / q) * q` where `X` is `log_normal`. `q` defaults to `1`. |

The proceeding tabs demonstrate how to specify different distributions in a YAML file for random or Bayesian search:

<Tabs
  defaultValue="constant"
  values={[
    {label: 'Constant', value: 'constant'},
    {label: 'Categorical', value: 'categorical'},
    {label: 'Uniform', value: 'uniform'},
    {label: 'Quantized log uniform', value: 'q_uniform'}
  ]}>
  <TabItem value="constant">

```yaml title="config.yaml"
parameter_name:
  distribution: constant
  value: 2.71828
```
  </TabItem>
  <TabItem value="categorical">

```yaml title="config.yaml"
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

```yaml title="config.yaml"
parameter_name:
  distribution: uniform
  min: 0
  max: 1
```
  </TabItem>
  <TabItem value="q_uniform">

```yaml title="config.yaml"
parameter_name:
  distribution: q_uniform
  min: 0
  max: 256
  q: 1
```
  </TabItem>
</Tabs>


## `metric`

Use the `metric` top-level sweep configuration key to specify the name, the goal, and the target metric to optimize.

|Key | Description |
| -------- | --------------------------------------------------------- |
| `name`   | Name of the metric to optimize.                           |
| `goal`   | Either `minimize` or `maximize` (Default is `minimize`).  |
| `target` | Goal value for the metric you're optimizing. When any run in the sweep achieves that target value, the sweep's state will be set to `finished`. This means all agents with active runs will finish those jobs, but no new runs will be launched in the sweep. |



## `parameters`

Specify one or more hyperparameters to explore during a sweep. 

In your YAML file or Python script, specify `parameters` as a top level key. Within the `parameters` key, provide the name of a hyperparameter you want to optimize. Common hyperparameters include: learning rate, batch size, epochs, optimizers, and more.

For each hyperparameter, specify one of the following:
* One or more values as a list
* A probability distribution and constraints on the sample space. See [Distribution options for random and Bayesian search](#distribution-options-for-random-and-bayesian-search) for a list of supported distributions. 

The proceeding table shows [INSERT]. See the code snippets that follow the proceeding code snippets for example sweep configurations that utilize various hyperparameter type, values, and distributions.


| Values          | Description   |
| --------------- | ------------------------------------------------------------------------------ |
| `values`        | Specifies all valid values for this hyperparameter. Compatible with `grid`.    |
| `value`         | Specifies the single valid value for this hyperparameter. Compatible with `grid`.  |
| `distribution`  | Specify a probability [distribution](#distribution-options-for-random-and-bayesian-search). See the note following this table for information on default values. |
| `probabilities` | Specify the probability of selecting each element of `values` when using `random`.  |
| `min`, `max`    | (`int`or `float`) Maximum and minimum values. If `int`, for `int_uniform` -distributed hyperparameters. If `float`, for `uniform` -distributed hyperparameters. |
| `mu`            | (`float`) Mean parameter for `normal` - or `lognormal` -distributed hyperparameters. |
| `sigma`         | (`float`) Standard deviation parameter for `normal` - or `lognormal` -distributed hyperparameters. |
| `q`             | (`float`) Quantization step size for quantized hyperparameters.     |
| `parameters`    | Nest other parameters inside a root level parameter.    |


:::note
If not specified, W&B will set distribution based on the following conditions:
* `categorical` if `values` is set
* `int_uniform` if `max` and `min` are set to integers
* `uniform` if `max` and `min` are set to floats
* `constant` if `value` is set
:::

The proceeding tabs show sweep configurations that [INSERT].


<details>
<summary>Sweep config with multiple hyperparameter values specified</summary>

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

</details>
<details>
<summary>Sweep config with single, multiple, and distribution hyperparameter values specified</summary>

```python title="train.py"
sweep_configuration = {
    "name": "sweepdemo",
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
</details>






<!-- 
<Tabs
  defaultValue="single"
  values={[
    {label: 'single value', value: 'single'},
    {label: 'multiple values', value: 'multiple'},
    {label: 'probabilities', value: 'probabilities'},
    {label: 'distribution', value: 'distribution'},
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
</Tabs> -->

## `early_terminate`

Use early termination (`early_terminate`) to stop poorly-performing runs. If early stopping is triggered, W&B will first stop the current run before it creates a new run with a new set of hyperparameter values. 

You must specify a stopping algorithm if you use early terminate. Nest the `type`key within `early_terminate` within your sweep configuration.

<Tabs
  defaultValue="cli"
  values={[
    {label: 'CLI', value: 'cli'},
    {label: 'Python script or Jupyter notebook', value: 'notebook'},
  ]}>
  <TabItem value="cli">

```yaml title="config.yaml"
early_terminate: 
    type: hyperband
```

  </TabItem>
  <TabItem value="notebook">

```python title="train.py"
sweep_config = {"early_terminate": {"type": hyperband}}
```

  </TabItem>
</Tabs>

W&B currently supports hyperband stopping algorithm. 


[Hyperband](https://arxiv.org/abs/1603.06560) hyperparameter optimization evaluates if a program should be stopped or permitted to continue at one or more pre-set iteration counts, called "brackets". When a run reaches a bracket, its metric value is compared to all previous reported metric values and the [W&B Run](../../ref/python/run.md) is terminated if its value is too high (when the goal is minimization) or low (when the goal is maximization).

Brackets are based on the number of logged iterations. The number of brackets corresponds to the number of times you log the metric you are optimizing. The iterations can correspond to steps, epochs, or something in between. The numerical value of the step counter is not used in bracket calculations.

:::info
Specify either `min_iter` or `max_iter` to create a bracket schedule.
:::


| Key        | Description                                                    |
| ---------- | -------------------------------------------------------------- |
| `min_iter` | Specify the iteration for the first bracket                    |
| `max_iter` | Specify the maximum number of iterations.                      |
| `s`        | Specify the total number of brackets (required for `max_iter`) |
| `eta`      | Specify the bracket multiplier schedule (default: `3`).        |
| `strict`   | Enable 'strict' mode that prunes runs aggressively, more closely following the original Hyperband paper. Defaults to false. |

The proceeding code snippets show example YAML sweep configurations that specify either a minimum or maximum number of iterations:

<Tabs
  defaultValue="min_iter"
  values={[
    {label: 'Minimum number of iterations specified', value: 'min_iter'},
    {label: 'Maximum number of iterations specified', value: 'max_iter'},
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

:::info
The hyperband early terminator checks what [W&B runs](../../ref/python/run.md) to terminate once every few minutes. The end run timestamp might differ from the specified brackets if your run or iteration are short.
:::

## `command` 

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

### Examples

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
