---
description: Hyperparameter search and model optimization with W&B Sweeps
menu:
  default:
    identifier: sweeps
    parent: w-b-models
title: Sweeps
url: guides/sweeps
weight: 2
cascade:
- url: guides/sweeps/:filename
---
{{< cta-button productLink="https://wandb.ai/stacey/deep-drive/workspace?workspace=user-lavanyashukla" colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb" >}}


Use W&B Sweeps to automate hyperparameter search and visualize rich, interactive experiment tracking. Pick from popular search methods such as Bayesian, grid search, and random to search the hyperparameter space. Scale and parallelize sweep across one or more machines.

{{< img src="/images/sweeps/intro_what_it_is.png" alt="Hyperparameter tuning insights" >}}

### How it works
Create a sweep with two [W&B CLI]({{< relref "/ref/cli/" >}}) commands:


1. Initialize a sweep

```bash
wandb sweep --project <project-name> <path-to-config file>
```

2. Start the sweep agent

```bash
wandb agent <sweep-ID>
```

{{% alert %}}
The preceding code snippet, and the colab linked on this page, show how to initialize and create a sweep with wht W&B CLI. See the [Sweeps walkthrough]({{< relref "./walkthrough.md" >}}) for a step-by-step outline of the W&B Python SDK commands to use to define a sweep configuration, initialize a sweep, and start a sweep.
{{% /alert %}}



### How to get started

Depending on your use case, explore the following resources to get started with W&B Sweeps:

* Read through the [sweeps walkthrough]({{< relref "./walkthrough.md" >}}) for a step-by-step outline of the W&B Python SDK commands to use to define a sweep configuration, initialize a sweep, and start a sweep.
* Explore this chapter to learn how to:
  * [Add W&B to your code]({{< relref "./add-w-and-b-to-your-code.md" >}})
  * [Define sweep configuration]({{< relref "/guides/models/sweeps/define-sweep-configuration/" >}})
  * [Initialize sweeps]({{< relref "./initialize-sweeps.md" >}})
  * [Start sweep agents]({{< relref "./start-sweep-agents.md" >}})
  * [Visualize sweep results]({{< relref "./visualize-sweep-results.md" >}})
* Explore a [curated list of Sweep experiments]({{< relref "./useful-resources.md" >}}) that explore hyperparameter optimization with W&B Sweeps. Results are stored in W&B Reports.

For a step-by-step video, see: [Tune Hyperparameters Easily with W&B Sweeps](https://www.youtube.com/watch?v=9zrmUIlScdY\&ab_channel=Weights%26Biases).

<!-- {% embed url="https://wandb.me/sweeps-video" %} -->