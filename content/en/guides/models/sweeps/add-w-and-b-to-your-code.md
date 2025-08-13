---
description: Add W&B to your Python code script or Jupyter Notebook.
menu:
  default:
    identifier: add-w-and-b-to-your-code
    parent: sweeps
title: Add W&B (wandb) to your code
weight: 2
---

This guide provides recommendations on how to add W&B Python APIs to your Python script or notebook.

## Original training script

Suppose you have a Python script that trains a model (see below). Your goal is to find the hyperparameters that maxmimizes the validation accuracy(`val_acc`).

In your Python script, you define two functions: `train_one_epoch` and `evaluate_one_epoch`. The `train_one_epoch` function simulates training for one epoch and returns the training accuracy and loss. The `evaluate_one_epoch` function simulates evaluating the model on the validation data set and returns the validation accuracy and loss.

You define a configuration dictionary (`config`) that contains hyperparameter values such as the learning rate (`lr`), batch size (`batch_size`), and number of epochs (`epochs`). The values in the configuration dictionary are used to control the training process. 

Next you define a function called `main` that mimics a typical training loop. For each epoch, the accuracy and loss is computed on the training and validation data sets.

{{< alert >}}
This code is a mock training script. It does not train a model, but simulates the training process by generating random accuracy and loss values. The purpose of this code is to demonstrate how to integrate W&B into your training script.
{{< /alert >}}

```python
import random
import numpy as np

def train_one_epoch(epoch, lr, batch_size):
    acc = 0.25 + ((epoch / 30) + (random.random() / 10))
    loss = 0.2 + (1 - ((epoch - 1) / 10 + random.random() / 5))
    return acc, loss

def evaluate_one_epoch(epoch):
    acc = 0.1 + ((epoch / 20) + (random.random() / 10))
    loss = 0.25 + (1 - ((epoch - 1) / 10 + random.random() / 6))
    return acc, loss

# config variable with hyperparameter values
config = {"lr": 0.0001, "batch_size": 16, "epochs": 5}

def main():
    lr = config["lr"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]

    for epoch in np.arange(1, epochs):
        train_acc, train_loss = train_one_epoch(epoch, lr, batch_size)
        val_acc, val_loss = evaluate_one_epoch(epoch)

        print("epoch: ", epoch)
        print("training accuracy:", train_acc, "training loss:", train_loss)
        print("validation accuracy:", val_acc, "validation loss:", val_loss)

if __name__ == "__main__":
    main()
```

In the next section, you will add W&B to your Python script to track hyperparameters and metrics during training. You want to use W&B to find the best hyperparameters that maximize the validation accuracy (`val_acc`).


## Training script with W&B Python SDK

How you integrate W&B to your Python script or notebook depends on how you manage sweeps. You can start a sweep job within a Python notebook or script or from the command line.

{{< tabpane text=true >}}
{{% tab header="Python script or notebook" %}} 

Add the following to your Python script:

1. Import the W&B Python SDK (`wandb`).
2. Create a dictionary object where the key-value pairs define a [sweep configuration]({{< relref "/guides/models/sweeps/define-sweep-configuration/" >}}). The sweep configuration defines the hyperparameters you want W&B to explore on your behalf along with the metric you want to optimize. Continuing from the previous example, the batch size (`batch_size`), epochs (`epochs`), and the learning rate (`lr`) are the hyperparameters to vary during each sweep. You want to maximize the accuracy of the validation score so you set `"goal": "maximize"` and the name of the variable you want to optimize for, in this case `val_acc` (`"name": "val_acc"`).
3. Pass the sweep configuration dictionary to [`wandb.sweep()`]({{< relref "/ref/python/sdk/functions/sweep.md" >}}). This initializes the sweep and returns a sweep ID (`sweep_id`). For more information, see [Initialize sweeps]({{< relref "./initialize-sweeps.md" >}}).
4. Use [`wandb.init()`]({{< relref "/ref/python/sdk/functions/init.md" >}}) to generate a background process to sync and log data as a [W&B Run]({{< relref "/ref/python/sdk/classes/run.md" >}}).
5. Log the metric you are optimizing for to W&B using [`wandb.Run.log()`]({{< relref "/ref/python/sdk/classes/run.md/#method-runlog" >}}). Within the configuration dictionary (`sweep_configuration` in this example), you define the sweep to maximize the `val_acc` value.
{{% alert %}}
You must log the metric you define and are optimizing for in both your sweep configuration and with `wandb.Run.log()`. For example, if you define the metric to optimize as `val_acc`, you must log `val_acc`. If you do not log the metric, W&B does not know what to optimize for.
{{% /alert %}}


6. Start the sweep with [`wandb.agent()`]({{< relref "/ref/python/sdk/functions/agent.md" >}}). Provide the sweep ID and the name of the function the sweep will execute (`function=main`), and specify the maximum number of runs to try to four (`count=4`). For more information, see [Start sweep agents]({{< relref "./start-sweep-agents.md" >}}).


Putting this all together, your script might look simlar to the following:

```python
import wandb # Import the W&B Python SDK
import numpy as np
import random

def train_one_epoch(epoch, lr, batch_size):
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
    # Metric that you want to optimize
    # For example, if you want to maximize validation
    # accuracy set "goal": "maximize" and the name of the variable 
    # you want to optimize for, in this case "val_acc"
    "metric": {
        "goal": "maximize", 
        "name": "val_acc"
        },
    "parameters": {
        "batch_size": {"values": [16, 32, 64]},
        "epochs": {"values": [5, 10, 15]},
        "lr": {"max": 0.1, "min": 0.0001},
    },
}

def main():
    # Initialize the sweep by passing in the config dictionary
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project)

    # Start the sweep job
    wandb.agent(sweep_id, function=main, count=4)

    # Use the `with` context manager statement to automatically end the run.
    with wandb.init() as run:
        # Fetches the hyperparameter values from `wandb.Run.config` object
        # instead of defining them explicitly
        lr = run.config["lr"]
        batch_size = run.config["batch_size"]
        epochs = run.config["epochs"]

        # Execute the training loop and log the performance values to W&B
        for epoch in np.arange(1, epochs):
            train_acc, train_loss = train_one_epoch(epoch, lr, batch_size)
            val_acc, val_loss = evaluate_one_epoch(epoch)

            run.log(
                {
                    "epoch": epoch,
                    "train_acc": train_acc,
                    "train_loss": train_loss,
                    "val_acc": val_acc, # Metric optimized
                    "val_loss": val_loss,
                }
            )

if __name__ == "__main__":
    main()
```



{{% /tab %}} {{% tab header="CLI" %}}

Create a YAML configuration file with your sweep configuration. The
configuration file contains the hyperparameters you want the sweep to explore. In
the following example, the batch size (`batch_size`), epochs (`epochs`), and
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

Next, add the following to the code example:

1. Import the W&B Python SDK (`wandb`) and PyYAML (`yaml`). PyYAML is used to read in our YAML configuration file.
2. Read in the configuration file.
3. Use the [`wandb.init()`]({{< relref "/ref/python/sdk/functions/init.md" >}}) API to generate a background process to sync and log data as a [W&B Run]({{< relref "/ref/python/sdk/classes/run.md" >}}). Pass the config object to the config parameter.
4. Define hyperparameter values from `wandb.Run.config` instead of using hard coded values.
5. Log the metric you want to optimize with [`wandb.Run.log()`]({{< relref "/ref/python/sdk/classes/run.md/#method-runlog" >}}). You must log the metric defined in your configuration. Within the configuration dictionary (`sweep_configuration` in this example) you define the sweep to maximize the `val_acc` value.

```python
import wandb
import yaml
import random
import numpy as np


def train_one_epoch(epoch, lr, batch_size):
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

    with wandb.init(config=config) as run:
        for epoch in np.arange(1, run.config['epochs']):
            train_acc, train_loss = train_one_epoch(epoch, run.config['lr'], run.config['batch_size'])
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

Copy the sweep ID and replace `sweepID` in the following code snippet to start
the sweep job with the [`wandb agent`]({{< relref "/ref/cli/wandb-agent.md" >}})
command:

```bash
wandb agent --count $NUM your-entity/sweep-demo-cli/sweepID
```

For more information, see [Start sweep jobs]({{< relref "./start-sweep-agents.md" >}}).

{{% /tab %}} {{< /tabpane >}}

## Logging metrics to W&B in a sweep

Log the sweep's metric to W&B explicitly. Do not log metrics for your sweep inside a subdirectory.

### Correctly logging metrics to W&B in a sweep

Explicitly access the key-value pair within the Python dictionary. For example, the following code specifies the key-value pair when you pass the dictionary to the `wandb.Run.log()` method:

```python title="train.py"
# Import the W&B Python Library and log into W&B
import wandb
import random

def train():
    epoch = 5  # Define epoch variable
    offset = random.random() / 5
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset

    return loss, acc

def main():
    with wandb.init(entity="<entity>", project="my-first-sweep") as run:
        # Correct
        run.log({"val_loss": val_loss, "val_acc": val_acc})

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

### Incorrectly logging metrics to W&B in a sweep

Consider the following pseudocode. A user wants to log the validation loss (`"val_loss": loss`). First they pass the values into a dictionary. However, the dictionary passed to `wandb.Run.log()` does not explicitly access the key-value pair in the dictionary. Instead, the user nests the dictionary inside another dictionary.

```python
# Import the W&B Python Library and log into W&B
import wandb
import random

def train():
    # Simulate training and validation metrics
    offset = random.random() / 5
    epoch = 5  # Simulate an epoch value
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset
    return loss, acc


def main():
    with wandb.init(entity="<entity>", project="my-first-sweep") as run:
        val_loss, val_acc = train()
        # Incorrect
        run.log({"validation": {"loss": val_loss, "acc": val_acc}})

sweep_configuration = {
    "method": "random",
    "metric": {"goal": "minimize", "name": "val_loss"},
    "parameters": {
        "x": {"max": 0.1, "min": 0.01},
        "y": {"values": [1, 3, 7]},
    },
}

# Initialize the sweep with the configuration dictionary
sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")

# Start the sweep job
wandb.agent(sweep_id, function=main, count=10)
```

