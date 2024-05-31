---
displayed_sidebar: default
---

# Manage job inputs

Add variable inputs to your job.

## Intro

Jobs have customizable inputs, enabling users to reconfigure the job when launching from the W&B CLI or UI.

By default W&B Jobs capture the entire Run.config as the inputs to a job. The launch sdk provides function to select keys in the run config, or specify JSON or YAML files as inputs.

For information on how job inputs can be set when launching from the CLI or UI, see the [Enqueue jobs](./add-job-to-queue.md) guide.

:::info
`launch` sdk functions require `wandb-core`. See the [`wandb-core` README](https://github.com/wandb/wandb/blob/main/core/README.md) for more information.
:::

## Reconfigure the `Run` object

The `Run` object returned by `wandb.init` in a job can be reconfigured, by default. The `launch` sdk provides a way to customize what parts of the `Run.config` object can be reconfigured when launching the job.


```python
import wandb
from wandb.sdk import launch

# Required for launch sdk use.
wandb.require("core")

config = {
    "trainer": {
        "learning_rate": 0.01,
        "batch_size": 32,
        "model": "resnet",
        "dataset": "cifar10",
        "private": {
            "key": "value",
        },
    },
    "seed": 42,
}


with wandb.init(config=config)
    launch.manange_wandb_config(
        include=["trainer"], 
        exclude=["trainer.private"],
    )
    # Etc.
```

The function `launch.manage_wandb_config` configures the job to accept input values for the `Run.config` object. The optional `include` and `exclude` options take path prefixes within the nested config object.

If `include` prefixes are provided, only paths within the config that match an `include` prefix will accept input values. If `exclude` prefixes are provided, no paths that match the `exclude` list will be filtered out of the input values. If a path matches both an `include` and an `exclude` prefix, the `exclude` prefix will take precedence.

In the preceding example, the path `["trainer.private"]` will filter out the `private` key from the `trainer` object, and the path `["trainer"]` will filter out all keys not under the `trainer` object.

:::tip
Use a `\`-escaped `.` to filter out keys with a `.` in their name. 

For example, `r"trainer\.private"` filters out the `trainer.private` key rather than the `private` key under the `trainer` object.

Note that the `r` prefix above denotes a raw string.
:::

If the code above is packaged and run as a job, the input types of the job will be:

```json
{
    "trainer": {
        "learning_rate": "float",
        "batch_size": "int",
        "model": "str",
        "dataset": "str",
    },
}
```

When launching the job from the W&B CLI or UI, the user will be able to override the `trainer` parameters.

### Access run config inputs

Jobs launched with run config inputs can access the input values through the `Run.config`. The `Run` returned by `wandb.init` in the job code will have the input values automatically set. Use 
```python
from wandb.sdk import launch

run_config_overrides = launch.load_wandb_config()
```
to load the run config input values anywhere in the job code.

## Reconfigure a file

The launch sdk can manage input values stored in config files in the job code.

The `launch` sdk provides a way to manage input values stored in config files in the job code. The `launch.manage_config_file` function can be used to add a config file as an input to the job and apply input values to the file at runtime.

By default, no run config inputs will be captured if `launch.manage_config_file` is used. Calling `launch.manage_wandb_config` overrides this behavior.

Consider the following example:

```python
import yaml
import wandb
from wandb.sdk import launch

# Required for launch sdk use.
wandb.require("core")

launch.manage_config_file("config.yaml")

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

with wandb.init(config=config)
    # Etc.
```

Imagine the code is run with an adjacent file `config.yaml`:

```yaml
learning_rate: 0.01
batch_size: 32
model: resnet
dataset: cifar10
```

The call to `launch.manage_config_file` will add the `config.yaml` file as an input to the job, making it reconfigurable when launching from the W&B CLI or UI. 

The `include` and `exclude` keyword arugments may be used to filter the acceptable input keys for the config file in the same way as `launch.manage_wandb_config`.


### Access config file inputs

When `launch.manage_config_file` is called in a run created by Launch, `launch` patches the contents of the config file with the input values. The patched config file is available in the job environment.

:::important
Call `launch.manage_config_file` before reading the config file in the job code to ensure input values are used.
:::