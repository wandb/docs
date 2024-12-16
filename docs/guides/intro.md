---
description: >-
  An overview of what is W&B along with links on how to get started
  if you are a first time user.
url: /guides
displayed_sidebar: default
title: What is W&B?
---

Weights & Biases (W&B) is the AI developer platform, with tools for training models, fine-tuning models, and leveraging foundation models. 

{{< img src="/images/general/architecture.png" alt="" >}}

W&B consists of three major components: [Models](/guides/models.md), [Weave](https://wandb.github.io/weave/), and [Core](/guides/core.md):

**[W&B Models](/guides/models.md)** is a set of lightweight, interoperable tools for machine learning practitioners training and fine-tuning models.
- [Experiments](/guides/track/intro.md): Machine learning experiment tracking
- [Sweeps](/guides/sweeps/intro.md): Hyperparameter tuning and model optimization
- [Registry](/guides/registry/intro.md): Publish and share your ML models and datasets

**[W&B Weave](https://wandb.github.io/weave/)** is a lightweight toolkit for tracking and evaluating LLM applications.

**[W&B Core](/guides/core.md)** is set of powerful building blocks for tracking and visualizing data and models, and communicating results.
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

<iframe width="100%" height="330" src="https://www.youtube.com/embed/tHAFujRhZLA" title="Weights &amp; Biases End-to-End Demo" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## Are you a first-time user of W&B?

Try the [quickstart](../quickstart.md) to learn how to install W&B and how to add W&B to your code.
