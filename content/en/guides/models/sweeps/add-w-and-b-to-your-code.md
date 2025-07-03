---
description: Add W&B to your Python code script or Jupyter Notebook.
menu:
  default:
    identifier: add-w-and-b-to-your-code
    parent: sweeps
title: Add W&B (wandb) to your code
weight: 2
---

There are numerous ways to add the W&B Python SDK to your script or notebook. This section provides a "best practice" example that shows how to integrate the W&B Python SDK into your own code.

### Original training script

Suppose you have the following code in a Python script. We define a function called `main` that mimics a typical training loop. For each epoch, the accuracy and loss is computed on the training and validation data sets. The values are randomly generated for the purpose of this example.

We defined a dictionary called `config` where we store hyperparameters values. At the end of the cell, we call the `main` function to execute the mock training code.

```python
import random
import numpy as np

def train_one_epoch(epoch, lr, bs):
    acc = 0.25 + ((epoch / 30) + (random.random() / 10))
    loss = 0.2 + (1 - ((epoch - 1) / 10 + random.random() / 5))
    return acc, loss

def evaluate_one_epoch(epoch):
    acc = 0.1 + ((epoch / 20) + (random.random() / 10))
    loss = 0.25 + (1 - ((epoch - 1) / 10 + random.random() / 6))
    return acc, loss

# config variable with hyperparameter values
config = {"lr": 0.0001, "bs": 16, "epochs": 5}

def main():
    # Note that we define values from `wandb.config`
    # instead of defining hard values
    lr = config["lr"]
    bs = config["bs"]
    epochs = config["epochs"]

    for epoch in np.arange(1, epochs):
        train_acc, train_loss = train_one_epoch(epoch, lr, bs)
        val_acc, val_loss = evaluate_one_epoch(epoch)

        print("epoch: ", epoch)
        print("training accuracy:", train_acc, "training loss:", train_loss)
        print("validation accuracy:", val_acc, "training loss:", val_loss)        
```

### Training script with W&B Python SDK

The following code examples demonstrate how to add the W&B Python SDK into your
code. If you start W&B Sweep jobs in the CLI, you will want to explore the CLI
tab. If you start W&B Sweep jobs within a Jupyter notebook or Python script,
explore the Python SDK tab.

{{< tabpane text=true >}} {{% tab header="Python script or notebook" %}} To
create a W&B Sweep, we added the following to the code example:

1. Import the Weights & Biases Python SDK.
2. Create a dictionary object where the key-value pairs define the sweep configuration. In the proceeding example, the batch size (`batch_size`), epochs (`epochs`), and the learning rate (`lr`) hyperparameters are varied during each sweep. For more information, see [Define sweep configuration]({{< relref "/guides/models/sweeps/define-sweep-configuration/" >}}).
3. Pass the sweep configuration dictionary to [`wandb.sweep`]({{< relref "/ref/python/sdk/functions/sweep.md" >}}). This initializes the sweep. This returns a sweep ID (`sweep_id`). For more information, see [Initialize sweeps]({{< relref "./initialize-sweeps.md" >}}).
4. Use the [`wandb.init()`]({{< relref "/ref/python/sdk/functions/init.md" >}}) API to generate a background process to sync and log data as a [W&B Run]({{< relref "/ref/python/sdk/classes/run.md" >}}).
5. (Optional) define values from `wandb.config` instead of defining hard coded values.
6. Log the metric you want to optimize with [`run.log`]({{< relref "/ref/python/sdk/classes/run.md/#method-runlog" >}}). You must log the metric defined in your configuration. Within the configuration dictionary (`sweep_configuration` in this example), you define the sweep to maximize the `val_acc` value.
7. Start the sweep with the [`wandb.agent`]({{< relref "/ref/python/sdk/functions/agent.md" >}}) API call. Provide the sweep ID and the name of the function the sweep will execute (`function=main`), and specify the maximum number of runs to try to four (`count=4`). For more informationp, see [Start sweep agents]({{< relref "./start-sweep-agents.md" >}}).


```python
import wandb
import numpy as np
import random


# Define training function that takes in hyperparameter
# values from `wandb.config` and uses them to train a
# model and return the metrics
def train_one_epoch(epoch, lr, bs):
    acc = 0.25 + ((epoch / 30) + (random.random() / 10))
    loss = 0.2 + (1 - ((epoch - 1) / 10 + random.random() / 5))
    return acc, loss


def evaluate_one_epoch(epoch):
    acc = 0.1 + ((epoch / 20) + (random.random() / 10))
    loss = 0.25 + (1 - ((epoch - 1) / 10 + random.random() / 6))
    return acc, loss


# Define a sweep config dictionary
sweep_configuration = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "val_acc"},
    "parameters": {
        "batch_size": {"values": [16, 32, 64]},
        "epochs": {"values": [5, 10, 15]},
        "lr": {"max": 0.1, "min": 0.0001},
    },
}

# (Optional) Provide a name for the project.
project = "my-first-sweep"

def main():
    # Use the `with` context manager statement to automatically end the run.
    # This is equivalent to using `run.finish()` at the end of each run
    with wandb.init(project=project) as run:

        # This code fetches the hyperparameter values from `wandb.config`
        # instead of defining them explicitly
        lr = run.config["lr"]
        bs = run.config["batch_size"]
        epochs = run.config["epochs"]

        # Execute the training loop and log the performance values to W&B
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
    # Initialize the sweep by passing in the config dictionary
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project)

    # Start the sweep job
    wandb.agent(sweep_id, function=main, count=4)

```

{{% alert %}} The preceding code snippet shows how to initialize a
[`wandb.init()`]({{< relref "/ref/python/sdk/functions/init.md" >}}) API within a `with`
context manager statement to generate a background process to sync and log data
as a [W&B Run]({{< relref "/ref/python/sdk/classes/run.md" >}}). This ensures the run is
properly terminated after uploading the logged values. An alternative approach
is to call `wandb.init()` and `wandb.finish()` at the beginning and end of the
training script, respectively.
{{% /alert %}}

{{% /tab %}} {{% tab header="CLI" %}}

To create a W&B Sweep, we first create a YAML configuration file. The
configuration file contains the hyperparameters we want the sweep to explore. In
the proceeding example, the batch size (`batch_size`), epochs (`epochs`), and
the learning rate (`lr`) hyperparameters are varied during each sweep.

```yaml
# config.yaml
program: train.py
method: random
name: sweep
metric:
  goal: maximize
  name: val_acc
parameters:
  batch_size:
    values: [16, 32, 64]
  lr:
    min: 0.0001
    max: 0.1
  epochs:
    values: [5, 10, 15]
```

For more information on how to create a W&B Sweep configuration, see [Define sweep configuration]({{< relref "/guides/models/sweeps/define-sweep-configuration/" >}}).

You must provide the name of your Python script for the `program` key
in your YAML file.

Next, we add the following to the code example:

1. Import the Weights & Biases Python SDK (`wandb`) and PyYAML (`yaml`). PyYAML is used to read in our YAML configuration file.
2. Read in the configuration file.
3. Use the [`wandb.init()`]({{< relref "/ref/python/sdk/functions/init.md" >}}) API to generate a background process to sync and log data as a [W&B Run]({{< relref "/ref/python/sdk/classes/run.md" >}}). We pass the config object to the config parameter.
4. Define hyperparameter values from `wandb.config` instead of using hard coded values.
5. Log the metric we want to optimize with [`wandb.log`]({{< relref "/ref/python/sdk/classes/run.md/#method-runlog" >}}). You must log the metric defined in your configuration. Within the configuration dictionary (`sweep_configuration` in this example) we defined the sweep to maximize the `val_acc` value.


```python
import wandb
import yaml
import random
import numpy as np


def train_one_epoch(epoch, lr, bs):
    acc = 0.25 + ((epoch / 30) + (random.random() / 10))
    loss = 0.2 + (1 - ((epoch - 1) / 10 + random.random() / 5))
    return acc, loss


def evaluate_one_epoch(epoch):
    acc = 0.1 + ((epoch / 20) + (random.random() / 10))
    loss = 0.25 + (1 - ((epoch - 1) / 10 + random.random() / 6))
    return acc, loss


def main():
    # Set up your default hyperparameters
    with open("./config.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    wandb.init(config=config)

    # Note that we define values from `wandb.config`
    # instead of  defining hard values
    lr = wandb.config.lr
    bs = wandb.config.batch_size
    epochs = wandb.config.epochs

    for epoch in np.arange(1, epochs):
        train_acc, train_loss = train_one_epoch(epoch, lr, bs)
        val_acc, val_loss = evaluate_one_epoch(epoch)

        wandb.log(
            {
                "epoch": epoch,
                "train_acc": train_acc,
                "train_loss": train_loss,
                "val_acc": val_acc,
                "val_loss": val_loss,
            }
        )


# Call the main function.
main()
```

In your CLI, set a maximum number of runs for the sweep
agent to try. This is optional. This example we set the
maximum number to 5.

```bash
NUM=5
```

Next, initialize the sweep with the [`wandb sweep`]({{< relref "/ref/cli/wandb-sweep.md" >}}) command. Provide the name of the YAML file. Optionally provide the name of the project for the project flag (`--project`):

```bash
wandb sweep --project sweep-demo-cli config.yaml
```

This returns a sweep ID. For more information on how to initialize sweeps, see
[Initialize sweeps]({{< relref "./initialize-sweeps.md" >}}).

Copy the sweep ID and replace `sweepID` in the proceeding code snippet to start
the sweep job with the [`wandb agent`]({{< relref "/ref/cli/wandb-agent.md" >}})
command:

```bash
wandb agent --count $NUM your-entity/sweep-demo-cli/sweepID
```

For more information, see [Start sweep jobs]({{< relref "./start-sweep-agents.md" >}}).

{{% /tab %}} {{< /tabpane >}}

## Consideration when logging metrics

Be sure to log the sweep's metric to W&B explicitly. Do not log metrics for your sweep inside a subdirectory.

For example, consider the proceeding pseudocode. A user wants to log the validation loss (`"val_loss": loss`). First they pass the values into a dictionary. However, the dictionary passed to `wandb.log` does not explicitly access the key-value pair in the dictionary:

```python
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
    # Incorrect. You must explicitly access the
    # key-value pair in the dictionary
    # See next code block to see how to correctly log metrics
    wandb.log({"val_loss": val_metrics})


sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters": {
        "x": {"max": 0.1, "min": 0.01},
        "y": {"values": [1, 3, 7]},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")

wandb.agent(sweep_id, function=main, count=10)
```

Instead, explicitly access the key-value pair within the Python dictionary. For example, the proceeding code specifies the key-value pair when you pass the dictionary to the `wandb.log` method:

```python title="train.py"
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
    wandb.log({"val_loss": val_metrics["val_loss"]})


sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters": {
        "x": {"max": 0.1, "min": 0.01},
        "y": {"values": [1, 3, 7]},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")

wandb.agent(sweep_id, function=main, count=10)
```
