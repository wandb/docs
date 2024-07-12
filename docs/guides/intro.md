---
description: >-
  An overview of what is W&B along with links on how to get started
  if you are a first time user.
slug: /guides
displayed_sidebar: default
---

# What is W&B?

Weights & Biases (W&B) is the AI developer platform, with tools for training models, fine-tuning models, and leveraging foundation models. 

![](@site/static/images/general/architecture.png)

W&B consists of three major components: [Models](/guides/models.md), [Weave](https://wandb.github.io/weave/), and [Core](/guides/platform.md):

**[W&B Models](/guides/models.md)** is a set of lightweight, interoperable tools for machine learning practitioners training and fine-tuning models.
- [Experiments](/guides/track/intro.md): Machine learning experiment tracking
- [Sweeps](/guides/sweeps/intro.md): Hyperparameter tuning and model optimization
- [Model Registry](/guides/model_registry/intro.md): Manage production models centrally
- [Launch](/guides/launch/intro.md): Scale and automate workloads

**[W&B Weave](https://wandb.github.io/weave/)** is a lightweight toolkit for tracking and evaluating LLM applications.

**[W&B Core](/guides/platform.md)** is set of powerful building blocks for tracking and visualizing data and models, and communicating results.
- [Artifacts](/guides/artifacts/intro.md): Version assets and track lineage
- [Tables](/guides/tables/intro.md): Visualize and query tabular data
- [Reports](/guides/reports/intro.md): Document and collaborate on your discoveries
<!-- - [Weave](/guides/app/features/panels/weave) Query and create visualizations of your data -->


## How does W&B work?



Read the following sections in this order if you are a first-time user of W&B and you are interested in training, tracking, and visualizing machine learning models and experiments:

1. Learn about [runs](./runs/intro.md), W&B's basic unit of computation.
2. Create and track machine learning experiments with [Experiments](./track/intro.md).
3. Discover W&B's flexible and lightweight building block for dataset and model versioning with [Artifacts](./artifacts/intro.md).
4. Automate hyperparameter search and explore the space of possible models with [Sweeps](./sweeps/intro.md).
5. Manage the model lifecycle from training to production with [Model Registry](./model_registry/intro.md).
6. Visualize predictions across model versions with our [Data Visualization](./tables/intro.md) guide.
7. Organize runs, embed and automate visualizations, describe your findings, and share updates with collaborators with [Reports](./reports/intro.md).


## Are you a first-time user of W&B?

Start exploring W&B with these resources:

1. [Quickstart](../quickstart.md): Install W&B and read a quick overview of how and where to add W&B to your code
1. [Intro notebook](http://wandb.me/intro): Learn how to train and track a machine learning experiment.
1. Explore the [Integrations guide](./integrations/intro.md) and the [W&B Easy Integration YouTube](https://www.youtube.com/playlist?list=PLD80i8An1OEGDADxOBaH71ZwieZ9nmPGC) playlist for information on how to integrate W&B with your preferred machine learning framework.
1. View the [API Reference guide](../ref/README.md) for technical specifications about the W&B Python Library, CLI, and Query Language operations.

<iframe width="100%" height="330" src="https://www.youtube.com/embed/tHAFujRhZLA" title="Weights &amp; Biases End-to-End Demo" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>