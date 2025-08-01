---
description: Track a model, the model's dependencies, and other information relevant
  to that model with the W&B Python SDK.
menu:
  default:
    identifier: log-model-to-experiment
    parent: model-registry
title: Track a model
weight: 3
---


Track a model, the model's dependencies, and other information relevant to that model with the W&B Python SDK. 

Under the hood, W&B creates a lineage of [model artifact]({{< relref "./model-management-concepts.md#model-artifact" >}}) that you can view with the W&B App or programmatically with the W&B Python SDK. See the [Create model lineage map]({{< relref "./model-lineage.md" >}}) for more information.

## How to log a model

Use the `run.log_model` API to log a model. Provide the path where your model files are saved to the `path` parameter. The path can be a local file, directory, or [reference URI]({{< relref "/guides/core/artifacts/track-external-files.md#amazon-s3--gcs--azure-blob-storage-references" >}}) to an external bucket such as `s3://bucket/path`. 

Optionally provide a name for the model artifact for the `name` parameter. If `name` is not specified, W&B uses the basename of the input path prepended with the run ID. 

Copy and paste the proceeding code snippet. Ensure to replace values enclosed in `<>` with your own.

```python
import wandb

# Initialize a W&B run
run = wandb.init(project="<project>", entity="<entity>")

# Log the model
run.log_model(path="<path-to-model>", name="<name>")
```

<details>

<summary>Example: Log a Keras model to W&B</summary>

The proceeding code example shows how to log a convolutional neural network (CNN) model to W&B.

```python
import os
import wandb
from tensorflow import keras
from tensorflow.keras import layers

config = {"optimizer": "adam", "loss": "categorical_crossentropy"}

# Initialize a W&B run
run = wandb.init(entity="charlie", project="mnist-project", config=config)

# Training algorithm
loss = run.config["loss"]
optimizer = run.config["optimizer"]
metrics = ["accuracy"]
num_classes = 10
input_shape = (28, 28, 1)

model = keras.Sequential(
    [
        layers.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

# Save model
model_filename = "model.h5"
local_filepath = "./"
full_path = os.path.join(local_filepath, model_filename)
model.save(filepath=full_path)

# Log the model
run.log_model(path=full_path, name="MNIST")

# Explicitly tell W&B to end the run.
run.finish()
```
</details>