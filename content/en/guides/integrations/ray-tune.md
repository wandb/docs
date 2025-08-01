---
description: How to integrate W&B with Ray Tune.
menu:
  default:
    identifier: ray-tune
    parent: integrations
title: Ray Tune
weight: 360
---

W&B integrates with [Ray](https://github.com/ray-project/ray) by offering two lightweight integrations.

- The`WandbLoggerCallback` function automatically logs metrics reported to Tune to the Wandb API.
- The `setup_wandb()` function, which can be used with the function API,  automatically initializes the Wandb API with Tune's training information. You can use the Wandb API as usual. such as by using `run.log()` to log your training process.

## Configure the integration

```python
from ray.air.integrations.wandb import WandbLoggerCallback
```

Wandb configuration is done by passing a wandb key to the config parameter of `tune.run()` (see example below).

The content of the wandb config entry is passed to `wandb.init()` as keyword arguments. The exception are the following settings, which are used to configure the `WandbLoggerCallback` itself:

### Parameters

`project (str)`: Name of the Wandb project. Mandatory.

`api_key_file (str)`: Path to file containing the Wandb API KEY.

`api_key (str)`: Wandb API Key. Alternative to setting `api_key_file`.

`excludes (list)`: List of metrics to exclude from the log.

`log_config (bool)`: Whether to log the config parameter of the results dictionary. Defaults to False.

`upload_checkpoints (bool)`:  If True, model checkpoints are uploaded as artifacts. Defaults to False.

### Example

```python
from ray import tune, train
from ray.air.integrations.wandb import WandbLoggerCallback


def train_fc(config):
    for i in range(10):
        train.report({"mean_accuracy": (i + config["alpha"]) / 10})


tuner = tune.Tuner(
    train_fc,
    param_space={
        "alpha": tune.grid_search([0.1, 0.2, 0.3]),
        "beta": tune.uniform(0.5, 1.0),
    },
    run_config=train.RunConfig(
        callbacks=[
            WandbLoggerCallback(
                project="<your-project>", api_key="<your-api-key>", log_config=True
            )
        ]
    ),
)

results = tuner.fit()
```

## setup_wandb

```python
from ray.air.integrations.wandb import setup_wandb
```

This utility function helps initialize Wandb for use with Ray Tune. For basic usage, call `setup_wandb()` in your training function:

```python
from ray.air.integrations.wandb import setup_wandb


def train_fn(config):
    # Initialize wandb
    wandb = setup_wandb(config)
    run = wandb.init(
        project=config["wandb"]["project"],
        api_key_file=config["wandb"]["api_key_file"],
    )

    for i in range(10):
        loss = config["a"] + config["b"]
        run.log({"loss": loss})
        tune.report(loss=loss)
    run.finish()


tuner = tune.Tuner(
    train_fn,
    param_space={
        # define search space here
        "a": tune.choice([1, 2, 3]),
        "b": tune.choice([4, 5, 6]),
        # wandb configuration
        "wandb": {"project": "Optimization_Project", "api_key_file": "/path/to/file"},
    },
)
results = tuner.fit()
```

## Example Code

We've created a few examples for you to see how the integration works:

* [Colab](https://wandb.me/raytune-colab): A simple demo to try the integration.
* [Dashboard](https://wandb.ai/anmolmann/ray_tune): View dashboard generated from the example.