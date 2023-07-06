---
slug: /guides/sweeps
description: Hyperparameter search and model optimization with W&B Sweeps
displayed_sidebar: default
---
import { CTAButtons } from '@site/src/components/CTAButtons/CTAButtons.tsx';


# Tune Hyperparameters

<CTAButtons productLink="" colabLink=""/>

<head>
  <title>Tune Hyperparameters with Sweeps</title>
</head>

Use W&B Sweeps to automate hyperparameter search and explore the space of possible models. Create a sweep with a few lines of code. Sweeps combines the benefits of automated hyperparameter search with our visualization-rich, interactive experiment tracking. Pick from popular search methods such as Bayesian, grid search, and random to search the hyperparameter space.  Scale and parallelize Sweep jobs across one or more machines.

![Draw insights from large hyperparameter tuning experiments with interactive dashboards.](/images/sweeps/intro_what_it_is.png)

### How it works
Create a sweep with two [W&B CLI](../../ref/cli/README.md) commands:


1. Initialize a sweep

```bash
wandb sweep --project <propject-name> <path-to-config file>
```

2. Start the sweep agent

```bash
wandb agent <sweep-ID>
```

:::tip
The preceding code snippet, and the colab linked on this page, show how to initialize and create a sweep with wht W&B CLI. See the Sweeps [Walkthrough] for a step-by-step outline of the W&B Python SDK commands to use to define a sweep configuration, initialize a sweep, and start a sweep.
:::



### How to get started

Depending on your use case, explore the following resources to get started with W&B Sweeps:

* If this is your first time using W&B Sweeps, we recommend you go through the Sweeps Colab notebook.
* Read through the [sweeps walkthrough](./quickstart.md) for a step-by-step outline of the W&B Python SDK commands to use to define a sweep configuration, initialize a sweep, and start a sweep.
* Explore this chapter to learn how to:
  * [Add W&B to your code](./add-w-and-b-to-your-code.md)
  * [Define sweep configuration](./define-sweep-configuration.md)
  * [Initialize sweeps](./initialize-sweeps.md)
  * [Start sweep agents](./start-sweep-agents.md)
  * [Visualize sweep results](./visualize-sweep-results.md)
* Explore a [curated list of Sweep experiments](./useful-resources.md) that explore hyperparameter optimization with W&B Sweeps. Results are stored in W&B Reports.

For a step-by-step video, see: [Tune Hyperparameters Easily with W&B Sweeps](https://www.youtube.com/watch?v=9zrmUIlScdY\&ab\_channel=Weights%26Biases).

<!-- {% embed url="http://wandb.me/sweeps-video" %} -->

