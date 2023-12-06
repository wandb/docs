---
displayed_sidebar: default
---

# Log models

The following guide describes how to log models to a W&B run and interact with them. 

:::tip
The following APIs are useful for tracking models as a part of your experiment tracking workflow. Use the APIs listed on this page to quickly log models to a run, in addition to metrics, tables, media and other objects.

W&B suggests that you use [W&B Artifacts](../../artifacts/intro.md) if you want to:
- Create and keep track of different versions of serialized data besides models, such as datasets, prompts, and more.
- Explore [lineage graphs](../../artifacts/explore-and-traverse-an-artifact-graph.md) of a model or any other objects tracked in W&B.
- How to interact with the model artifacts these methods created, such as [updating properties](../../artifacts/update-an-artifact.md) (metadata, aliases, and descriptions) 

For more information on W&B Artifacts and for more information on advanced versioning use cases, see the [Artifacts](../../artifacts/intro.md) documentation.
:::

:::info
See this [Colab notebook](https://colab.research.google.com/drive/1Nvgz4VQHMbr4hoVGeUdDVfHE2lHFpvbs) for an end-to-end example of how to use the APIs described on this page.
:::

## Log a model to a W&B run
Use the [`log_model`](../../../ref/python/run.md#logmodel) to log a model artifact that contains content within a directory you specify. The `log_model` methods also marks the resulting model artifact as an output of the W&B run. 

Models that are associated to a W&B run enable you to track the lineage of that model. You can view the lineage of the model, such as the inputs and outputs of a run, within the W&B App UI. See the [Explore and traverse artifact graphs](../../artifacts/explore-and-traverse-an-artifact-graph.md) page within the [Artifacts](../../artifacts/intro.md) chapter for more information.

Provide the path where your model file(s) are saved to the `path` parameter. The path can be a local file, directory, or [reference URI](../../artifacts/track-external-files.md#amazon-s3--gcs--azure-blob-storage-references) to an external bucket such as `s3://bucket/path`. Optionally provide a name for the model artifact for the `name` parameter. If `name` is not specified, W&B will use the basename of the input path prepended with the run ID as the name. 

Ensure to replace values enclosed in `<>` with your own.

```python
import wandb

# Initialize a W&B run
run = wandb.init(project="<your-project>", entity="<your-entity>")

# Log the model
run.log_model(path="<path-to-model>")
```

See [`log_model`](../../../ref/python/run.md#logmodel) in the API Reference guide for more information on possible parameters.

<details>

<summary>Example: Log a model to a run</summary>

In the proceeding code snippet, a path to the model file `/local/dir/70154.h5` is provided to the `path` parameter.

```python
import wandb

path = "/local/dir/70154.h5"
model_artifact_name = "fine-tuned-model"

# Initialize a W&B run
run = wandb.init(project="MNIST_Exploration", entity="charlie")

# Log the model
run.log_model(path=path, name=model_artifact_name)
run.finish()
```

When the user called `log_model`, a model artifact with name `fine-tuned-model` was created and the file `70154.h5` was added to the model artifact.

</details>


## Download and use a logged model
Use the [`use_model`](../../../ref/python/run.md#usemodel) function to access and download models files previously logged to a W&B run. 

Provide the name of the model artifact where the model file(s) you are looking to retrieve are stored for the `name` parameter. The name you provide must match the name of an existing logged model artifact. The name must adhere to one of the following schemas: 

* `model_artifact_name:version`
* `model_artifact_name:alias`
* `model_artifact_name:digest`

:::info
You can optionally prepend the the W&B entity and project to the name. For example, the following is valid:

```python
run.use_model(
            name="<entity>/<project>/<model_artifact_name>:<digest>",
        )
```
:::

If you did not define `name` when originally logged the file(s) with `log_model`, the default name assigned is the basename of the input path, prepended with the run ID.

Ensure to replace other the values enclosed in `<>` with your own:
 
```python
import wandb

# Initialize a run
run = wandb.init(project="<your-project>", entity="<your-entity>")

# Access and download model. Returns path to downloaded artifact
downloaded_model_path = run.use_model(name="<your-model-name>")
```

The `use_model` function returns the path of downloaded artifact file(s). Keep track of this path if you want to link this model later. In the preceding code snippet, the returned path is stored in a variable called `downloaded_model_path`.

<details>

<summary>Example: Download and use a logged model</summary>

For example, the proceeding code snippet shows how to log a model with `log_model` method. First, the user defines a `model_name` variable that contains the full name of the model artifact. Then the user called the `use_model` API to access and download the model. They then stored the path that is returned from the API to the `downloaded_model_path` variable.

```python
import wandb

entity="luka"
project="NLP_Experiments"
alias="latest"
model_artifact_name = "fine-tuned-model"
model_name = f"{entity}/{project}/{model_artifact_name}:{alias}"

# Initialize a run
run = wandb.init(project=project, entity=entity)

# Access and download model. Returns path to downloaded artifact
downloaded_model_path = run.use_model(name=model_name)
```
</details>

See [`use_model`](../../../ref/python/run.md#usemodel) in the API Reference guide for more information on possible parameters and return type.

## Log and link a model to the W&B Model Registry
Use the [`link_model`](../../../ref/python/run.md#linkmodel) method to log model file(s) to a W&B run and link it to the [W&B Model Registry](../../model_registry/intro.md). If no registered model exists, W&B will create a new for you with the name you provide for the `linked_model_name` parameter. 

:::tip
You can think of linking a model similar to 'bookmarking' or 'publishing' a model that others members of your team can view.
:::

A *Registered Model* is a collection or folder of linked model versions in the W&B Model Registry. Registered models typically represent a team’s ML task. 

The proceeding code snippet shows how to link a model with the `link_model` API. Ensure to replace other the values enclosed in `<>` with your own:

```python
import wandb

run = wandb.init(entity="<your-entity>", project="<your-project>")

run.link_model(
    path="<path-to-model>",
    registered_model_name="<model-registry-name>"
)
```

See [`link_model`](../../../ref/python/run.md#linkmodel) in the API Reference guide for more information on optional parameters.

A new version of a registered model is created when you link a model artifact to a model registry that already exists within that model registry.

For example, suppose you have a  model artifact named "mnist-testing" that exists within a model registry called "MNIST". And suppose that within the W&B App UI you see that the model artifact is marked as **Version 1**.  W&B will automatically create a **Version 2** of your model if you link a new model with the same name ("mnist-testing") to the same registry ("MNIST").

<details>

<summary>Example: Log and link a model to the W&B Model Registry</summary>

For example, the proceeding code snippet logs model files and links the model model to a model registry called `"MNIST"`. 

To do this, a user calls the `link_model` API. When they call the API, they provide a local filepath that points the content of the model (`path`) and they provide a name for the model registry (`registered_model_name`). 

```python
import wandb

path = "/local/dir/model.pt"
registered_model_name = "MNIST"

run = wandb.init(project="<your-project>", entity="<your-entity>")

run.link_model(
    path=path,
    registered_model_name=registered_model_name
)
```

</details>