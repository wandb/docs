# Keras




[![](https://www.tensorflow.org/images/GitHub-Mark-32px.png)View source on GitHub](https://www.github.com/wandb/client/tree/c4726707ed83ebb270a2cf84c4fd17b8684ff699/wandb/integration/keras/__init__.py)



Tools for integrating `wandb` with [`Keras`](https://keras.io/).


Keras is a deep learning API for [`TensorFlow`](https://www.tensorflow.org/).

## Classes

[`class WandbCallback`](./wandbcallback.md): `WandbCallback` automatically integrates keras with wandb.

[`class WandbEvalCallback`](./wandbevalcallback.md): Abstract base class to build Keras callbacks for model prediction visualization.

[`class WandbMetricsLogger`](./wandbmetricslogger.md): Logger that sends system metrics to W&B.

[`class WandbModelCheckpoint`](./wandbmodelcheckpoint.md): A checkpoint that periodically saves a Keras model or model weights.

