---
description: Track machine learning experiments with W&B.
slug: /guides/track
---

# Track Experiments

<head>
  <title>Track Machine Learning and Deep Learning Experiments.</title>
</head>

Use the W&B Python Library to track machine learning experiments with a few lines of code. You can then review the results in an [interactive dashboard](app.md) or export your data to Python for programmatic access using our [Public API](https://github.com/wandb/gitbook/tree/9daa732ca79ab1f56edf77631db3bdb259e0b3c5/guides/track/advanced/public-api-guide.md). Quickly find and [reproduce machine learning experiments with W&B](./reproduce-experiments.md).

Utilize W&B Integrations if you use use popular frameworks such as [PyTorch](../integrations/pytorch.md), [Keras](../integrations/keras.md), or [Scikit](../integrations/scikit.md). See our [Integration guides](../integrations/intro.md) for a for a full list of integrations and information on how to add W&B to your code.

## How it works

W&B Experiments are composed of the following building blocks:

1. [**`wandb.init()`**](./launch.md): Initialize a new run at the top of your script. This returns a `Run` object and creates a local directory where all logs and files are saved, then streamed asynchronously to a W&B server. If you want to use a private server instead of our hosted cloud server, we offer [Self-Hosting](../hosting/intro.md).
2. [**`wandb.config`**](./config.md): Save a dictionary of hyperparameters such as learning rate or model type. The model settings you capture in config are useful later to organize and query your results.
3. [**`wandb.log()`**](./log/intro.md): Log metrics over time in a training loop, such as accuracy and loss. By default, when you call `wandb.log` it appends a new step to the `history` object and updates the `summary` object.
   * `history`: An array of dictionary-like objects that tracks metrics over time. These time series values are shown as default line plots in the UI.
   * `summary`: By default, the final value of a metric logged with wandb.log(). You can set the summary for a metric manually to capture the highest accuracy or lowest loss instead of the final value. These values are used in the table, and plots that compare runs — for example, you could visualize at the final accuracy for all runs in your project.
4. [**`wandb.log_artifact`**](../../ref/python/artifact.md): Save outputs of a run, like the model weights or a table of predictions. This lets you track not just model training, but all the pipeline steps that affect the final model.

The proceeding psuedocode demonstrates a common W&B Experiment tracking workflow:

```python
# Flexible integration for any Python script
import wandb

# 1. Start a W&B Run
wandb.init(project='my-project-name')

# 2. Save mode inputs and hyperparameters
config = wandb.config
config.learning_rate = 0.01

# Set up model and data
model, dataloader = get_model(), get_data()

# Model training goes here

# 3. Log metrics over time to visualize performance
wandb.log({"loss": loss})

# 4. Log an artifact to W&B
wandb.log_artifact(model)

```

## How to get started

If this is your first time using W&B Experiments, we recommend you read the Quick Start. The [Quickstart](../../quickstart.md) walks you through the steps to set up your first experiment. 


<!-- ## Storage options
You can sync data on premises (on-prem), in a private cloud, or a local instance of Weights and Biases.

* **On-Prem**: If you need a private cloud or local instance of W&B, see our [Self Hosted](../hosting/intro.md) offerings.
* **Automated Environments**: Most of these settings can also be controlled via [Environment Variables](./advanced/environment-variables.md). This is often useful when you're running jobs on a cluster. -->



