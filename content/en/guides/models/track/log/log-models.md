---
menu:
  default:
    identifier: log-models
    parent: log-objects-and-media
title: Log models
---

{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/ken-add-new-model-reg-api/colabs/wandb-model-registry/New_Model_Logging_in_W&B.ipynb" >}}
# Log models

The following guide describes how to log models to a W&B run and interact with them. 

{{% alert %}}
The following APIs are useful for tracking models as a part of your experiment tracking workflow. Use the APIs listed on this page to log models to a run, and to access metrics, tables, media, and other objects.

W&B suggests that you use [W&B Artifacts]({{< relref "/guides/core/artifacts/" >}}) if you want to:
- Create and keep track of different versions of serialized data besides models, such as datasets, prompts, and more.
- Explore [lineage graphs]({{< relref "/guides/core/artifacts/explore-and-traverse-an-artifact-graph.md" >}}) of a model or any other objects tracked in W&B.
- Interact with the model artifacts these methods created, such as [updating properties]({{< relref "/guides/core/artifacts/update-an-artifact.md" >}}) (metadata, aliases, and descriptions) 

For more information on W&B Artifacts and advanced versioning use cases, see the [Artifacts]({{< relref "/guides/core/artifacts/" >}}) documentation.
{{% /alert %}}

## Log a model to a run
Use the [`log_model`]({{< relref "/ref/python/sdk/classes/run.md#log_model" >}}) to log a model artifact that contains content within a directory you specify. The [`log_model`]({{< relref "/ref/python/sdk/classes/run.md#log_model" >}}) method also marks the resulting model artifact as an output of the W&B run. 

You can track a model's dependencies and the model's associations if you mark the model as the input or output of a W&B run. View the lineage of the model within the W&B App UI. See the [Explore and traverse artifact graphs]({{< relref "/guides/core/artifacts/explore-and-traverse-an-artifact-graph.md" >}}) page within the [Artifacts]({{< relref "/guides/core/artifacts/" >}}) chapter for more information.

Provide the path where your model files are saved to the `path` parameter. The path can be a local file, directory, or [reference URI]({{< relref "/guides/core/artifacts/track-external-files.md#amazon-s3--gcs--azure-blob-storage-references" >}}) to an external bucket such as `s3://bucket/path`. 

Ensure to replace values enclosed in `<>` with your own.

```python
import wandb

# Initialize a W&B run
run = wandb.init(project="<your-project>", entity="<your-entity>")

# Log the model
run.log_model(path="<path-to-model>", name="<name>")
```

Optionally provide a name for the model artifact for the `name` parameter. If `name` is not specified, W&B will use the basename of the input path prepended with the run ID as the name. 

{{% alert %}}
Keep track of the `name` that you, or W&B assigns, to the model. You will need the name of the model to retrieve the model path with the [`use_model`]({{< relref "/ref/python/sdk/classes/run.md#use_model" >}}) method. 
{{% /alert %}}

See [`log_model`]({{< relref "/ref/python/sdk/classes/run.md#log_model" >}}) in the API Reference for parameters.

<details>

<summary>Example: Log a model to a run</summary>

```python
import os
import wandb
from tensorflow import keras
from tensorflow.keras import layers

config = {"optimizer": "adam", "loss": "categorical_crossentropy"}

# Initialize a W&B run
run = wandb.init(entity="charlie", project="mnist-experiments", config=config)

# Hyperparameters
loss = run.config["loss"]
optimizer = run.config["optimizer"]
metrics = ["accuracy"]
num_classes = 10
input_shape = (28, 28, 1)

# Training algorithm
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

# Configure the model for training
model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

# Save model
model_filename = "model.h5"
local_filepath = "./"
full_path = os.path.join(local_filepath, model_filename)
model.save(filepath=full_path)

# Log the model to the W&B run
run.log_model(path=full_path, name="MNIST")
run.finish()
```

When the user called `log_model`, a model artifact named `MNIST` was created and the file `model.h5` was added to the model artifact. Your terminal or notebook will print information of where to find information about the run the model was logged to.

```python
View run different-surf-5 at: https://wandb.ai/charlie/mnist-experiments/runs/wlby6fuw
Synced 5 W&B file(s), 0 media file(s), 1 artifact file(s) and 0 other file(s)
Find logs at: ./wandb/run-20231206_103511-wlby6fuw/logs
```

</details>


## Download and use a logged model
Use the [`use_model`]({{< relref "/ref/python/sdk/classes/run.md#use_model" >}}) function to access and download models files previously logged to a W&B run. 

Provide the name of the model artifact where the model files you are want to retrieve are stored. The name you provide must match the name of an existing logged model artifact.

If you did not define `name` when originally logged the files with `log_model`, the default name assigned is the basename of the input path, prepended with the run ID.

Ensure to replace other the values enclosed in `<>` with your own:
 
```python
import wandb

# Initialize a run
run = wandb.init(project="<your-project>", entity="<your-entity>")

# Access and download model. Returns path to downloaded artifact
downloaded_model_path = run.use_model(name="<your-model-name>")
```

The [use_model]({{< relref "/ref/python/sdk/classes/run.md#use_model" >}}) function returns the path of downloaded model files. Keep track of this path if you want to link this model later. In the preceding code snippet, the returned path is stored in a variable called `downloaded_model_path`.

<details>

<summary>Example: Download and use a logged model</summary>

For example, in the proceeding code snippet a user called the `use_model` API. They specified the name of the model artifact they want to fetch and they also provided a version/alias. They then stored the path that is returned from the API to the `downloaded_model_path` variable.

```python
import wandb

entity = "luka"
project = "NLP_Experiments"
alias = "latest"  # semantic nickname or identifier for the model version
model_artifact_name = "fine-tuned-model"

# Initialize a run
run = wandb.init(project=project, entity=entity)
# Access and download model. Returns path to downloaded artifact
downloaded_model_path = run.use_model(name = f"{model_artifact_name}:{alias}") 
```
</details>

See [`use_model`]({{< relref "/ref/python/sdk/classes/run.md#use_model" >}}) in the API Reference for parameters and return type.

## Log and link a model to the W&B Model Registry

{{% alert %}}
The [`link_model`]({{< relref "/ref/python/sdk/classes/run.md#link_model" >}}) method is currently only compatible with the legacy W&B Model Registry, which will soon be deprecated. To learn how to link a model artifact to the new edition of model registry, visit the [Registry linking guide]({{< relref "/guides/core/registry/link_version.md" >}}). 
{{% /alert %}}

Use the [`link_model`]({{< relref "/ref/python/sdk/classes/run.md#link_model" >}}) method to log model files to a W&B Run and link it to the [W&B Model Registry]({{< relref "/guides/core/registry/model_registry/" >}}). If no registered model exists, W&B will create a new one for you with the name you provide for the `registered_model_name` parameter. 

Linking a model is analogous to 'bookmarking' or 'publishing' a model to a centralized team repository of models that others members of your team can view and consume. 

When you link a model, that model is not duplicated in the [Registry]({{< relref "/guides/core/registry/model_registry/" >}}) or moved out of the project and into the registry. A linked model is a pointer to the original model in your project.

Use the [Registry]({{< relref "/guides/core/registry/" >}}) to organize your best models by task, manage model lifecycle, facilitate easy tracking and auditing throughout the ML lifecyle, and [automate]({{< relref "/guides/core/automations/" >}}) downstream actions with webhooks or jobs.

A *Registered Model* is a collection or folder of linked model versions in the [Model Registry]({{< relref "/guides/core/registry/model_registry/" >}}). Registered models typically represent candidate models for a single modeling use case or task. 

The proceeding code snippet shows how to link a model with the [`link_model`]({{< relref "/ref/python/sdk/classes/run.md#link_model" >}}) API. Ensure to replace other the values enclosed in `<>` with your own:

```python
import wandb

run = wandb.init(entity="<your-entity>", project="<your-project>")
run.link_model(path="<path-to-model>", registered_model_name="<registered-model-name>")
run.finish()
```

See [`link_model`]({{< relref "/ref/python/sdk/classes/run.md#link_model" >}}) in the API Reference guide for optional parameters.

If the `registered-model-name` matches the name of a registered model that already exists within the Model Registry, the model will be linked to that registered model. If no such registered model exists, a new one will be created and the model will be the first one linked. 

For example, suppose you have an existing registered model named "Fine-Tuned-Review-Autocompletion" in your Model Registry (see example [here](https://wandb.ai/reviewco/registry/model?selectionPath=reviewco%2Fmodel-registry%2FFinetuned-Review-Autocompletion&view=all-models)). And suppose that a few model versions are already linked to it: v0, v1, v2. If you call `link_model` with `registered-model-name="Fine-Tuned-Review-Autocompletion"`, the new model will be linked to this existing registered model as v3. If no registered model with this name exists, a new one will be created and the new model will be linked as v0. 


<details>

<summary>Example: Log and link a model to the W&B Model Registry</summary>

For example, the proceeding code snippet logs model files and links the model to a registered model name `"Fine-Tuned-Review-Autocompletion"`. 

To do this, a user calls the `link_model` API. When they call the API, they provide a local filepath that points the content of the model (`path`) and they provide a name for the registered model to link it to (`registered_model_name`). 

```python
import wandb

path = "/local/dir/model.pt"
registered_model_name = "Fine-Tuned-Review-Autocompletion"

run = wandb.init(project="llm-evaluation", entity="noa")
run.link_model(path=path, registered_model_name=registered_model_name)
run.finish()
```

{{% alert %}}
Reminder: A registered model houses a collection of bookmarked model versions. 
{{% /alert %}}

</details>