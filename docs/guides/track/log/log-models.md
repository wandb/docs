---
displayed_sidebar: default
---

# Log models

Use W&B to log models to your experiment.

The following guide describes how to log and interact with models logged to a W&B run. 

:::tip
The following APIs are useful for tracking models as a part of your experiment tracking workflow. Use the APIs listed on this page to quickly log models to a run, in addition to metrics, tables, media and other objects.

W&B suggests that you use [W&B Artifacts](../../artifacts/intro.md) if you want to:
- Create and keep track of different versions of serialized data besides models, such as datasets, prompts, and more.
- Explore [lineage graphs](../../artifacts/explore-and-traverse-an-artifact-graph.md) of a model or any other objects tracked in W&B.
- How to interact with the model artifacts these methods created, such as [updating properties](../../artifacts/update-an-artifact.md) (metadata, aliases, and descriptions) 

For more information on W&B Artifacts and for more information on advanced versioning use cases, see the [Artifacts](../../artifacts/intro.md) documentation.
:::


## Log a model to a W&B run
Use the [`log_model`](../../../ref/python/run.md#logmodel) method to log a model artifact. The `log_model` methods also marks the model artifact as an output of the run. Associating a model to a run (with log_model, for example), enables you to track the lineage of the model. View the lineage of the model, such as the inputs and outputs of a run, within the W&B App UI. See the [Explore and traverse artifact graphs](../../artifacts/explore-and-traverse-an-artifact-graph.md) page within the [Artifacts](../../artifacts/intro.md) chapter for more information.

Provide a name for your model artifact and the path where your model is saved to for the `model_name` and `path` parameters, respectively. Ensure to replace values enclosed in `<>` with your own.

```python
import wandb

# Initialize a W&B run
run = wandb.init(project='<your-project>', entity='<your-entity>')

# Log the model
run.log_model(model_name='<model_artifact_name>', path='<path-to-model>')
run.finish()
```

The path can be a local file, directory, or [reference URI](../../artifacts/track-external-files.md#amazon-s3--gcs--azure-blob-storage-references) to an external bucket such as `s3://bucket/path`. 

<details>

<summary>Example: Log a model to a run</summary>

In the proceeding code snippet, a path to the model file(s) `/local/dir/70154.h5` is passed in.  When the user logged the model with `log_model`, they gave the model artifact a name of `model.h5`. 

```python
import wandb

project="<your-project-name>"
entity="<your-entity>"
path="/local/dir/70154.h5"
model_artifact_name="model.h5"

# Initialize a W&B run
run = wandb.init(project=project, entity=entity)

# Log the model
run.log_model(model_name=model_artifact_name, path=path)
run.finish()
```
</details>


## Download and use a logged model
Use the [`use_model`](../../../ref/python/run.md#usemodel) function to access and download models files previously logged to a W&B run. 

Provide the name of your model artifact to the `model_name` field for the `use_model` parameter. Ensure to replace other the values enclosed in `<>` with your own:

:::tip
W&B suggests that you prepend the entity and name of the project your model was saved to.
:::

 
```python
import wandb

# Initialize a run
run = wandb.init(project='<your-project>', entity='<your-entity>')

# Access and download model. Returns path to downloaded artifact
downloaded_model_path = run.use_model(model_name="<your-model-name>")
```

The `use_model` function returns the path of downloaded artifact file(s). Keep track of this path, as you will need to have this path to link a model. In the preceeding code snippet, we stored the file path in a variable called `downloaded_model_path`.

<details>

<summary>Example: Download and use a logged model</summary>

For example, the proceeding code snippet shows how to log a model with `log_model` method. First, the user defines a `model_name` variable that contains the full name of the model artifact. Then the user called the use_model API to access and download the model. They then stored the path that is returned from the API to the `downloaded_model_path` variable.


```python
import wandb

alias = "v0"
model_name=f'{entity}/{project}/{model_artifact_name}:{alias}'

# Initialize a run
run = wandb.init(project=project, entity=entity)

# Access and download model. Returns path to downloaded artifact
downloaded_model_path = run.use_model(model_name=model_name)
```

:::note
The code shown in this example is a continuation of the code example shown in the dropdown of the [Log a model to a W&B run](#log-a-model-to-a-wb-run) section. The code in this example uses the same `entity`, `project`, and `model_artifact_name` variables declared.
:::

</details>


## Log and link a model to the W&B Model Registry
Use the `link_model` method to log model file(s) as a model [artifact](../../artifacts/intro.md) to a W&B run and link it to the [W&B Model Registry](../../model_registry/intro.md). 

The proceeding code snippet shows how to link a model with the `link_model` API. Ensure to replace other the values enclosed in `<>` with your own:

```python
run.link_model(path=downloaded_model_path,
                 registered_model_name="<model-registry-name>",
                 linked_model_name="<linked-model=name>",
                 aliases=["<aliases>"])

run.finish()
```

:::tip
W&B will create a model registry with the name you provide for `linked_model_name` parameter if you do not already have a registry with that name.
:::


A new version of a registered model is created when you link a model artifact to a model registry that already exists within that model registry.

For example, suppose you have a  model artifact named "mnist-testing" that exists within a model registry called "MNIST". And suppose that within the W&B App UI you see that the model artifact is marked as **Version 1**.  W&B will automatically create a **Version 2** of your model if you link a new model with the same name ("mnist-testing") to the same registry ("MNIST").

<details>

<summary>Example: Log and link a model to the W&B Model Registry</summary>

For example, the proceeding code snippet links the model created in previous code snippets to a W&B Model Registry called `"MNIST"`. To do this, the user called the `link_model` API and provided the path of the downloaded model artifact, the name of the model registry the user wanted to link the model to, the name of the model, and an alias `"best"` for the `path`, `linked_model_name`, `model_name`, and `aliases` parameters, respectively. 

```python
registered_model_name="MNIST"

run.link_model(
    path=downloaded_model_path, 
    linked_model_name=registered_model_name,
    model_name=model_artifact_name, 
    aliases=['best'])
```

:::note
The code shown in this example is a continuation of the code example shown in the dropdown of the [Download and use a logged model](#download-and-use-a-logged-model) section. The code in this example uses the same `downloaded_model_path` and `model_artifact_name` variables declared.
:::


</details>