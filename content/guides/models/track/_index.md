---
description: Track machine learning experiments with W&B.
menu:
  default:
    identifier: experiments
    parent: w-b-models
title: Experiments
url: guides/track
weight: 1
cascade:
- url: guides/track/:filename
---
{{< cta-button productLink="https://wandb.ai/stacey/deep-drive/workspace?workspace=user-lavanyashukla" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases.ipynb" >}}

Track machine learning experiments with a few lines of code. You can then review the results in an [interactive dashboard]({{< relref "/guides/models/track/workspaces.md" >}}) or export your data to Python for programmatic access using our [Public API]({{< relref "/ref/python/public-api/" >}}). 

Utilize W&B Integrations if you use popular frameworks such as [PyTorch]({{< relref "/guides/integrations/pytorch.md" >}}), [Keras]({{< relref "/guides/integrations/keras.md" >}}), or [Scikit]({{< relref "/guides/integrations/scikit.md" >}}). See our [Integration guides]({{< relref "/guides/integrations/" >}}) for a for a full list of integrations and information on how to add W&B to your code.

{{< img src="/images/experiments/experiments_landing_page.png" alt="" >}}

The image above shows an example dashboard where you can view and compare metrics across multiple [runs]({{< relref "/guides/models/track/runs/" >}}).

## How it works

Track a machine learning experiment with a few lines of code:
1. Create a [W&B run]({{< relref "/guides/models/track/runs/" >}}).
2. Store a dictionary of hyperparameters, such as learning rate or model type, into your configuration ([`run.config`]({{< relref "./config.md" >}})).
3. Log metrics ([`run.log()`]({{< relref "./log/" >}})) over time in a training loop, such as accuracy and loss.
4. Save outputs of a run, like the model weights or a table of predictions.

The following code demonstrates a common W&B Experiment tracking workflow:

```python showLineNumbers
# 1. Start a W&B Run
#
# When this block exits, it waits for logged data to finish uploading.
# If an exception is raised, the run is marked failed.
with wandb.init(entity="", project="my-project-name") as run:

  # 2. Save mode inputs and hyperparameters
  run.config.learning_rate = 0.01

  # Import model and data
  model, dataloader = get_model(), get_data()

  # Model training code goes here

  # 3. Log metrics over time to visualize performance
  run.log({"loss": loss})

  # 4. Log an artifact to W&B
  run.log_artifact(model)
```

## How to get started

Depending on your use case, explore the following resources to get started with W&B Experiments:

* Read the [W&B Quickstart]({{< relref "/guides/quickstart.md" >}}) for a step-by-step outline of the W&B Python SDK commands you could use to create, track, and use a dataset artifact.
* Explore this chapter to learn how to:
  * Create an experiment
  * Configure experiments
  * Log data from experiments
  * View results from experiments
* Explore the [W&B Python Library]({{< relref "/ref/python/" >}}) within the [W&B API Reference Guide]({{< relref "/ref/" >}}).

## Best practices and tips 

For best practices and tips for experiments and logging, see [Best Practices: Experiments and Logging](https://wandb.ai/wandb/pytorch-lightning-e2e/reports/W-B-Best-Practices-Guide--VmlldzozNTU1ODY1#w&b-experiments-and-logging).