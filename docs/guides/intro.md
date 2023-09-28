---
description: >-
  An overview of what is W&B along with links on how to get started
  if you are a first time user.
slug: /guides
displayed_sidebar: default
---

# What is W&B?

Weights & Biases (W&B) is the AI developer platform, with tools for training models, fine-tuning models, and leveraging foundation models. Set up W&B in 5 minutes, then quickly iterate on your machine learning pipeline with the confidence that your datasets and models are tracked and versioned in a reliable system of record.

![](@site/static/images/general/product_architecture.png)

[W&B Models](/guides/models.md) is a set of lightweight, interoperable tools for machine learning practitioners training and fine-tuning models.
- [Experiments](/guides/track/intro.md): Machine learning experiment tracking
- [Model Registry](/guides/model_registry/intro.md): quickly track experiments
- [Launch](/guides/launch/intro.md): Scale and automate workloads
- [Sweeps](/guides/sweeps/intro.md): Hyperparameter tuning and model optimization

[W&B Prompts](/guides/prompts_platform)


## Are you a first-time user of W&B?

If this is your first time using W&B we suggest you explore the following:

1. Experience W&B in action, [run an example introduction project with Google Colab](http://wandb.me/intro).
1. Read through the [Quickstart](../quickstart.md) for a quick overview of how and where to add W&B to your code.
1. Read [How does W&B work?](#how-does-weights--biases-work) This section provides an overview of the building blocks of W&B.
1. Explore our [Integrations guide](./integrations/intro.md) and our [W&B Easy Integration YouTube](https://www.youtube.com/playlist?list=PLD80i8An1OEGDADxOBaH71ZwieZ9nmPGC) playlist for information on how to integrate W&B with your preferred machine learning framework.
1. View the [API Reference guide](../ref/README.md) for technical specifications about the W&B Python Library, CLI, and Weave operations.

## How does W&B work?

We recommend you read the following sections in this order if you are a first-time user of W&B:

1. Learn about [Runs](./runs/intro.md), W&B's basic unit of computation.
2. Create and track machine learning experiments with [Experiments](./track/intro.md).
3. Discover W&B's flexible and lightweight building block for dataset and model versioning with [Artifacts](./artifacts/intro.md).
4. Automate hyperparameter search and explore the space of possible models with [Sweeps](./sweeps/intro.md).
5. Manage the model lifecycle from training to production with [Model Management](./model_registry/intro.md).
6. Visualize predictions across model versions with our [Data Visualization](./tables/intro.md) guide.
7. Organize W&B Runs, embed and automate visualizations, describe your findings, and share updates with collaborators with [Reports](./reports/intro.md).
