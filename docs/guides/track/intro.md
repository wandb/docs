---
description: Track machine learning experiments with W&B.
slug: /guides/track
displayed_sidebar: default
---
import Translate, {translate} from '@docusaurus/Translate';
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';

# Track Experiments

<CTAButtons productLink="https://github.com/wandb/docodile" colabLink="https://www.google.com/"/>

<head>
  <title>Track Machine Learning and Deep Learning Experiments.</title>
</head>

Track machine learning experiments with a few lines of code. You can then review the results in an [interactive dashboard](app.md) or export your data to Python for programmatic access using our [Public API](../../ref/python/public-api/README.md). 

Utilize W&B Integrations if you use use popular frameworks such as [PyTorch](../integrations/pytorch.md), [Keras](../integrations/keras.md), or [Scikit](../integrations/scikit.md). See our [Integration guides](../integrations/intro.md) for a for a full list of integrations and information on how to add W&B to your code.

[INSERT photo]

## How it works

<!-- W&B Experiments are composed of the following building blocks: -->

<!-- 1. [**`wandb.init()`**](./launch.md): Initialize a new run at the top of your script. This returns a `Run` object and creates a local directory where all logs and files are saved, then streamed asynchronously to a W&B server. If you want to use a private server instead of our hosted cloud server, we offer [Self-Hosting](../hosting/intro.md).
2. [**`wandb.config`**](./config.md): Save a dictionary of hyperparameters such as learning rate or model type. The model settings you capture in config are useful later to organize and query your results.
3. [**`wandb.log()`**](./log/intro.md): Log metrics over time in a training loop, such as accuracy and loss. By default, when you call `wandb.log` it appends a new step to the `history` object and updates the `summary` object.
   * `history`: An array of dictionary-like objects that tracks metrics over time. These time series values are shown as default line plots in the UI.
   * `summary`: By default, the final value of a metric logged with wandb.log(). You can set the summary for a metric manually to capture the highest accuracy or lowest loss instead of the final value. These values are used in the table, and plots that compare runs — for example, you could visualize at the final accuracy for all runs in your project.
4. [**`wandb.log_artifact`**](../../ref/python/artifact.md): Save outputs of a run, like the model weights or a table of predictions. This lets you track not just model training, but all the pipeline steps that affect the final model. -->

Track a machine learning experiment with a few lines of code:
1. Create a [W&B run](../runs/intro.md).
2. Store a dictionary of hyperparameters, such as learning rate or model type, into your configuration ([`wandb.config`](./config.md)).
3. Log metrics ([`wandb.log()`](./log/intro.md)) over time in a training loop, such as accuracy and loss.
4. Save outputs of a run, like the model weights or a table of predictions.

The proceeding pseudocode demonstrates a common W&B Experiment tracking workflow:

```python showLineNumbers
# 1. Start a W&B Run
wandb.init(entity='', project='my-project-name')

# 2. Save mode inputs and hyperparameters
wandb.config.learning_rate = 0.01

# Import model and data
model, dataloader = get_model(), get_data()

# Model training code goes here

# 3. Log metrics over time to visualize performance
wandb.log({"loss": loss})

# 4. Log an artifact to W&B
wandb.log_artifact(model)
```

## How to get started

Depending on your use case, explore the following resources to get started with W&B Experiments:

* If this is your first time using W&B Artifacts, we recommend you go through the [Experiments Colab notebook](https://www.google.com/).
* Read the [W&B Quickstart](../../quickstart.md) for a step-by-step outline of the W&B Python SDK commands you could use to create, track, and use a dataset artifact.
* Explore this chapter to learn how to:
  * Create an experiment
  * Configure experiments
  * Log data from experiments
  * View results from experiments
* Explore the [W&B Python Library](../../ref/python/README.md) within the [W&B API Reference Guide](../../ref/README.md). 



