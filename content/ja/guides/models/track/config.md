---
description: Use a dictionary-like object to save your experiment configuration
menu:
  default:
    identifier: ja-guides-models-track-config
    parent: experiments
title: Configure experiments
weight: 2
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-log/Configs_in_W%26B.ipynb" >}}

Use the `config` property of a run to save your training configuration: 
- hyperparameter
- input settings such as the dataset name or model type
- any other independent variables for your experiments. 

The `run.config` property makes it easy to analyze your experiments and reproduce your work in the future. You can group by configuration values in the W&B App, compare the configurations of different W&B runs, and evaluate how each training configuration affects the output. The `config` property is a dictionary-like object that can be composed from multiple dictionary-like objects.

{{% alert %}}
To save output metrics or dependent variables like loss and accuracy, use `run.log` instead of `run.config`.
{{% /alert %}}



## Set up an experiment configuration
Configurations are typically defined in the beginning of a training script. Machine learning workflows may vary, however, so you are not required to define a configuration at the beginning of your training script.

Use dashes (`-`) or underscores (`_`) instead of periods (`.`) in your config variable names.
	
Use the dictionary access syntax `["key"]["value"]` instead of the attribute access syntax `config.key.value` if your script accesses `run.config` keys below the root.

The following sections outline different common scenarios of how to define your experiments configuration.

### Set the configuration at initialization
Pass a dictionary at the beginning of your script when you call the `wandb.init()` API to generate a background process to sync and log data as a W&B Run.

The proceeding code snippet demonstrates how to define a Python dictionary with configuration values and how to pass that dictionary as an argument when you initialize a W&B Run.

```python
import wandb

# Define a config dictionary object
config = {
    "hidden_layer_sizes": [32, 64],
    "kernel_sizes": [3],
    "activation": "ReLU",
    "pool_sizes": [2],
    "dropout": 0.5,
    "num_classes": 10,
}

# Pass the config dictionary when you initialize W&B
with wandb.init(project="config_example", config=config) as run:
    ...
```

If you pass a nested dictionary as the `config`, W&B flattens the names using dots.

Access the values from the dictionary similarly to how you access other dictionaries in Python:

```python
# Access values with the key as the index value
hidden_layer_sizes = run.config["hidden_layer_sizes"]
kernel_sizes = run.config["kernel_sizes"]
activation = run.config["activation"]

# Python dictionary get() method
hidden_layer_sizes = run.config.get("hidden_layer_sizes")
kernel_sizes = run.config.get("kernel_sizes")
activation = run.config.get("activation")
```

{{% alert %}}
Throughout the Developer Guide and examples we copy the configuration values into separate variables. This step is optional. It is done for readability.
{{% /alert %}}

### Set the configuration with argparse
You can set your configuration with an argparse object. [argparse](https://docs.python.org/3/library/argparse.html), short for argument parser, is a standard library module in Python 3.2 and above that makes it easy to write scripts that take advantage of all the flexibility and power of command line arguments.

This is useful for tracking results from scripts that are launched from the command line.

The proceeding Python script demonstrates how to define a parser object to define and set your experiment config. The functions `train_one_epoch` and `evaluate_one_epoch` are provided to simulate a training loop for the purpose of this demonstration:

```python
# config_experiment.py
import argparse
import random

import numpy as np
import wandb


# Training and evaluation demo code
def train_one_epoch(epoch, lr, bs):
    acc = 0.25 + ((epoch / 30) + (random.random() / 10))
    loss = 0.2 + (1 - ((epoch - 1) / 10 + random.random() / 5))
    return acc, loss


def evaluate_one_epoch(epoch):
    acc = 0.1 + ((epoch / 20) + (random.random() / 10))
    loss = 0.25 + (1 - ((epoch - 1) / 10 + random.random() / 6))
    return acc, loss


def main(args):
    # Start a W&B Run
    with wandb.init(project="config_example", config=args) as run:
        # Access values from config dictionary and store them
        # into variables for readability
        lr = run.config["learning_rate"]
        bs = run.config["batch_size"]
        epochs = run.config["epochs"]

        # Simulate training and logging values to W&B
        for epoch in np.arange(1, epochs):
            train_acc, train_loss = train_one_epoch(epoch, lr, bs)
            val_acc, val_loss = evaluate_one_epoch(epoch)

            run.log(
                {
                    "epoch": epoch,
                    "train_acc": train_acc,
                    "train_loss": train_loss,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                }
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "-e", "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "-lr", "--learning_rate", type=int, default=0.001, help="Learning rate"
    )

    args = parser.parse_args()
    main(args)
```
### Set the configuration throughout your script
You can add more parameters to your config object throughout your script. The proceeding code snippet demonstrates how to add new key-value pairs to your config object:

```python
import wandb

# Define a config dictionary object
config = {
    "hidden_layer_sizes": [32, 64],
    "kernel_sizes": [3],
    "activation": "ReLU",
    "pool_sizes": [2],
    "dropout": 0.5,
    "num_classes": 10,
}

# Pass the config dictionary when you initialize W&B
with wandb.init(project="config_example", config=config) as run:
    # Update config after you initialize W&B
    run.config["dropout"] = 0.2
    run.config.epochs = 4
    run.config["batch_size"] = 32
```

You can update multiple values at a time:

```python
run.config.update({"lr": 0.1, "channels": 16})
```

### Set the configuration after your Run has finished
Use the [W&B Public API]({{< relref path="/ref/python/public-api/" lang="ja" >}}) to update a completed run's config. 

You must provide the API with your entity, project name and the run's ID. You can find these details in the Run object or in the [W&B App UI]({{< relref path="/guides/models/track/workspaces.md" lang="ja" >}}):

```python
with wandb.init() as run:
    ...

# Find the following values from the Run object if it was initiated from the
# current script or notebook, or you can copy them from the W&B App UI.
username = run.entity
project = run.project
run_id = run.id

# Note that api.run() returns a different type of object than wandb.init().
api = wandb.Api()
api_run = api.run(f"{username}/{project}/{run_id}")
api_run.config["bar"] = 32
api_run.update()
```




## `absl.FLAGS`


You can also pass in [`absl` flags](https://abseil.io/docs/python/guides/flags).

```python
flags.DEFINE_string("model", None, "model to run")  # name, default, help

run.config.update(flags.FLAGS)  # adds absl flags to config
```

## File-Based Configs
If you place a file named `config-defaults.yaml` in the same directory as your run script, the run automatically picks up the key-value pairs defined in the file and passes them to `run.config`. 

The following code snippet shows a sample `config-defaults.yaml` YAML file:

```yaml
batch_size:
  desc: Size of each mini-batch
  value: 32
```

You can override the default values automatically loaded from `config-defaults.yaml` by setting updated values in the `config` argument of `wandb.init`. For example:

```python
import wandb

# Override config-defaults.yaml by passing custom values
with wandb.init(config={"epochs": 200, "batch_size": 64}) as run:
    ...
```

To load a configuration file other than `config-defaults.yaml`, use the `--configs command-line` argument and specify the path to the file:

```bash
python train.py --configs other-config.yaml
```

### Example use case for file-based configs
Suppose you have a YAML file with some metadata for the run, and then a dictionary of hyperparameters in your Python script. You can save both in the nested `config` object:

```python
hyperparameter_defaults = dict(
    dropout=0.5,
    batch_size=100,
    learning_rate=0.001,
)

config_dictionary = dict(
    yaml=my_yaml_file,
    params=hyperparameter_defaults,
)

with wandb.init(config=config_dictionary) as run:
    ...
```

## TensorFlow v1 flags

You can pass TensorFlow flags into the `wandb.config` object directly.

```python
with wandb.init() as run:
    run.config.epochs = 4

    flags = tf.app.flags
    flags.DEFINE_string("data_dir", "/tmp/data")
    flags.DEFINE_integer("batch_size", 128, "Batch size.")
    run.config.update(flags.FLAGS)  # add tensorflow flags as config
```