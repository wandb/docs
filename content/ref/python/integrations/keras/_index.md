---
title: Keras
---
<!-- Insert buttons and diff -->
{{< cta-button githubLink="https://www.github.com/wandb/wandb/tree/v0.18.7/wandb/integration/keras/__init__.py" >}}

Tools for integrating `wandb` with [`Keras`](https://keras.io/).

## Classes

[`class WandbCallback`](./wandbcallback.md): `WandbCallback` automatically integrates keras with wandb.

[`class WandbEvalCallback`](./wandbevalcallback.md): Abstract base class to build Keras callbacks for model prediction visualization.

[`class WandbMetricsLogger`](./wandbmetricslogger.md): Logger that sends system metrics to W&B.

[`class WandbModelCheckpoint`](./wandbmodelcheckpoint.md): A checkpoint that periodically saves a Keras model or model weights.
