---
title: "Do I need to provide values for all hyperparameters as part of the W&B Sweep. Can I set defaults?"
toc_hide: true
type: docs
tags:
   - sweeps
---
Access hyperparameter names and values from the sweep configuration using `wandb.config`, which acts like a dictionary.

For runs outside a sweep, set `wandb.config` values by passing a dictionary to the `config` argument in `wandb.init`. In a sweep, any configuration supplied to `wandb.init` serves as a default value, which the sweep can override.

Use `config.setdefaults` for explicit behavior. The following code snippets illustrate both methods:

{{< tabpane text=true >}}
{{% tab "wandb.init()" %}}
```python
# Set default values for hyperparameters
config_defaults = {"lr": 0.1, "batch_size": 256}

# Start a run and provide defaults
# that a sweep can override
with wandb.init(config=config_defaults) as run:
    # Add training code here
    ...
```
{{% /tab %}}
{{% tab "config.setdefaults()" %}}
```python
# Set default values for hyperparameters
config_defaults = {"lr": 0.1, "batch_size": 256}

# Start a run
with wandb.init() as run:
    # Update any values not set by the sweep
    run.config.setdefaults(config_defaults)

    # Add training code here
```
{{% /tab %}}
{{< /tabpane >}}