---
title: Keras
---
<!-- Insert buttons and diff -->
{{< cta-button githubLink="https://www.github.com/wandb/wandb/tree/v0.19.2/wandb/integration/keras/__init__.py" >}}

Tools for integrating `wandb` with [`Keras`](https://keras.io/).

## Classes

[`class WandbCallback`](./wandbcallback/): `WandbCallback` automatically integrates keras with wandb.

[`class WandbEvalCallback`](./wandbevalcallback/): Abstract base class to build Keras callbacks for model prediction visualization.

[`class WandbMetricsLogger`](./wandbmetricslogger/): Logger that sends system metrics to W&B.

[`class WandbModelCheckpoint`](./wandbmodelcheckpoint/): A checkpoint that periodically saves a Keras model or model weights.
