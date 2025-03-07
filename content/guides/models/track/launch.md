---
description: Create a W&B Experiment.
menu:
  default:
    identifier: launch
    parent: experiments
weight: 1
title: Create an experiment
---

Use the W&B Python SDK to track machine learning experiments. You can then review the results in an interactive dashboard or export your data to Python for programmatic access with the [W&B Public API]({{< relref "/ref/python/public-api/" >}}).

This guide describes how to use W&B building blocks to create a W&B Experiment. 

## How to create a W&B Experiment

Create a W&B Experiment in four steps:

1. [Initialize a W&B Run]({{< relref "#initialize-a-wb-run" >}})
2. [Capture a dictionary of hyperparameters]({{< relref "#capture-a-dictionary-of-hyperparameters" >}})
3. [Log metrics inside your training loop]({{< relref "#log-metrics-inside-your-training-loop" >}})
4. [Log an artifact to W&B]({{< relref "#log-an-artifact-to-wb" >}})

### Initialize a W&B run
At the beginning of your script call, the [`wandb.init()`]({{< relref "/ref/python/init.md" >}}) API to generate a background process to sync and log data as a W&B Run. 

The proceeding code snippet demonstrates how to create a new W&B project named `“cat-classification”`. A note `“My first experiment”` was added to help identify this run. Tags `“baseline”` and `“paper1”` are included to remind us that this run is a baseline experiment intended for a future paper publication.

```python
# Import the W&B Python Library
import wandb

# 1. Start a W&B Run
run = wandb.init(
    project="cat-classification",
    notes="My first experiment",
    tags=["baseline", "paper1"],
)
```
A [Run]({{< relref "/ref/python/run.md" >}}) object is returned when you initialize W&B with `wandb.init()`. Additionally, W&B creates a local directory where all logs and files are saved and streamed asynchronously to a W&B server.

{{% alert %}}
Note: Runs are added to pre-existing projects if that project already exists when you call wandb.init().  For example, if you already have a project called `“cat-classification”`, that project will continue to exist and not be deleted. Instead, a new run is added to that project.
{{% /alert %}}

### Capture a dictionary of hyperparameters
Save a dictionary of hyperparameters such as learning rate or model type. The model settings you capture in config are useful later to organize and query your results.

```python
#  2. Capture a dictionary of hyperparameters
wandb.config = {"epochs": 100, "learning_rate": 0.001, "batch_size": 128}
```
For more information on how to configure an experiment, see [Configure Experiments]({{< relref "./config.md" >}}).

### Log metrics inside your training loop
Log metrics during each `for` loop (epoch), the accuracy and loss values are computed and logged to W&B with [`wandb.log()`]({{< relref "/ref/python/log.md" >}}). By default, when you call wandb.log it appends a new step to the history object and updates the summary object.

The following code example shows how to log metrics with `wandb.log`.

{{% alert %}}
Details of how to set up your mode and retrieve data are omitted. 
{{% /alert %}}

```python
# Set up model and data
model, dataloader = get_model(), get_data()

for epoch in range(wandb.config.epochs):
    for batch in dataloader:
        loss, accuracy = model.training_step()
        #  3. Log metrics inside your training loop to visualize
        # model performance
        wandb.log({"accuracy": accuracy, "loss": loss})
```
For more information on different data types you can log with W&B, see [Log Data During Experiments]({{< relref "./log/" >}}).

### Log an artifact to W&B 
Optionally log a W&B Artifact. Artifacts make it easy to version datasets and models. 
```python
wandb.log_artifact(model)
```
For more information about Artifacts, see the [Artifacts Chapter]({{< relref "/guides/core/artifacts/" >}}). For more information about versioning models, see [Model Management]({{< relref "/guides/core/registry/model_registry/" >}}).


### Putting it all together
The full script with the preceding code snippets is found below:
```python
# Import the W&B Python Library
import wandb

# 1. Start a W&B Run
run = wandb.init(project="cat-classification", notes="", tags=["baseline", "paper1"])

#  2. Capture a dictionary of hyperparameters
wandb.config = {"epochs": 100, "learning_rate": 0.001, "batch_size": 128}

# Set up model and data
model, dataloader = get_model(), get_data()

for epoch in range(wandb.config.epochs):
    for batch in dataloader:
        loss, accuracy = model.training_step()
        #  3. Log metrics inside your training loop to visualize
        # model performance
        wandb.log({"accuracy": accuracy, "loss": loss})

# 4. Log an artifact to W&B
wandb.log_artifact(model)

# Optional: save model at the end
model.to_onnx()
wandb.save("model.onnx")
```

## Next steps: Visualize your experiment 
Use the W&B Dashboard as a central place to organize and visualize results from your machine learning models. With just a few clicks, construct rich, interactive charts like [parallel coordinates plots]({{< relref "/guides/models/app/features/panels/parallel-coordinates.md" >}}),[ parameter importance analyzes]({{< relref "/guides/models/app/features/panels/parameter-importance.md" >}}), and [more]({{< relref "/guides/models/app/features/panels/" >}}).

{{< img src="/images/sweeps/quickstart_dashboard_example.png" alt="Quickstart Sweeps Dashboard example" >}}

For more information on how to view experiments and specific runs, see [Visualize results from experiments]({{< relref "/guides/models/track/workspaces.md" >}}).


## Best practices
The following are some suggested guidelines to consider when you create experiments:

1. **Config**: Track hyperparameters, architecture, dataset, and anything else you'd like to use to reproduce your model. These will show up in columns— use config columns to group, sort, and filter runs dynamically in the app.
2. **Project**: A project is a set of experiments you can compare together. Each project gets a dedicated dashboard page, and you can easily turn on and off different groups of runs to compare different model versions.
3. **Notes**: Set a quick commit message directly from your script. Edit and access notes in the Overview section of a run in the W&B App.
4. **Tags**: Identify baseline runs and favorite runs. You can filter runs using tags. You can edit tags at a later time on the Overview section of your project's dashboard on the W&B App.
5. **Create multiple run sets to compare experiments**: When comparing experiments, create multiple run sets to make metrics easy to compare. You can toggle run sets on or off on the same chart or group of charts.

The following code snippet demonstrates how to define a W&B Experiment using the best practices listed above:

```python
import wandb

config = dict(
    learning_rate=0.01, momentum=0.2, architecture="CNN", dataset_id="cats-0192"
)

wandb.init(
    project="detect-cats",
    notes="tweak baseline",
    tags=["baseline", "paper1"],
    config=config,
)
```

For more information about available parameters when defining a W&B Experiment, see the [`wandb.init`]({{< relref "/ref/python/init.md" >}}) API docs in the [API Reference Guide]({{< relref "/ref/python/" >}}).