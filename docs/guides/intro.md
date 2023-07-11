---
description: >-
  Overview of the Weights & Biases machine learning developer platform, and 
slug: /guides
displayed_sidebar: default
---

# Weights & Biases

Use the W&B machine learning platform to build better models faster. 

Set up W&B in 5 minutes, then quickly iterate on your ML pipeline with the confidence that experiments, models, and datasets are tracked in a reliable system of record.

<iframe width="100%" height="330" src="https://www.youtube.com/embed/tHAFujRhZLA" title="Weights &amp; Biases End-to-End Demo" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>


## Start tracking models
- **[Colab Notebook](http://wandb.me/intro)**: Track your first experiment with W&B


## W&B Products
- **[Experiments](./track/intro.md)**: Track ML model training
- **[Reports](./reports/intro.md)**: Share findings with collaborators
- **[Artifacts](./artifacts/intro.md)**: Track assets, such as datasets or models
- **[Tables](./reports/intro.md)**: Visualize data and model predictions
- **[Sweeps](./reports/intro.md)**: Optimize hyperparameters
- **[Models](./reports/intro.md)**: Productionize workflows with a model registry
- **[Launch](./reports/intro.md)**: Connect to compute locally or in the cloud
- **[Prompts](./reports/intro.md)**: Visualize LLMs


![](@site/static/images/general/wandb_diagram_july23.png)

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
5. Manage the model lifecycle from training to production with [Model Management](./models/intro.md).
6. Visualize predictions across model versions with our [Data Visualization](./data-vis/intro.md) guide.
7. Organize W&B Runs, embed and automate visualizations, describe your findings, and share updates with collaborators with [Reports](./reports/intro.md).
