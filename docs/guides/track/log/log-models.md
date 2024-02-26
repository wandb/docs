---
displayed_sidebar: default
---

# Log models

Use W&B to log models that are created from W&B runs.

The following guide describes how to log and interact with models logged to a W&B run. 

:::tip
The following APIs are useful for early model exploration and experiment tracking. Use the APIs listed on this page to quickly log models to a run, in addition to metrics, tables, media and other objects.

The APIs listed on this page are not intended for [INSERT].


W&B suggests that you use W&B Artifacts if you want to:
- Create and keep track of different versions of serialized data such as models, datasets, prompts, and more.
- Create lineage graphs of a model or any other objects tracked in W&B.
- How to interact with the model artifacts these methods created, such as updating properties (metadata, aliases, and descriptions) 
:::


## Log a model to a W&B run
Declare a model artifact as an output of a run with the [`log_model`](../../../ref/python/run.md#logmodel) method. To do so, provide a name for your model artifact and the path where your model is saved to for the `model_name` and `path` parameters, respectively.

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

In the preceding code snippet, the model originally had a file name of `70154.h5` and was locally stored in the user's `/local/dir/` directory. When the user logged the model with `log_model`, they gave the model artifact a name of `model.h5`. 


## Download and use a logged model
Use the [`use_model`](../../../ref/python/run.md#usemodel) function to access and download models files previously logged to a W&B run. 

Provide the name of your model artifact to the `model_name` field in `use_model`. 

:::tip
W&B suggests that you prepend the entity and name of the project your model was saved to.
:::

The proceeding code snippet shows how to download a logged model. The code snippet uses the same variables declared in the [Log a model to a W&B run](#log-a-model-to-a-wb-run).

```python
import wandb

alias = "v0"
model_name=f'{entity}/{project}/{model_artifact_name}:{alias}'

# Initialize a run
run = wandb.init(project=project, entity=entity)

# Access and download model. Returns path to downloaded artifact
downloaded_model_path = run.use_model(model_name=model_name)
```

The `use_model` function returns the path of downloaded artifact file(s). Keep track of this path, as you will need to have this path to link a model. In the preceeding code snippet, we stored the file path in a variable called `downloaded_model_path`.


## Log and link a model to the W&B Model Registry
Use the `link_model` method to log model file(s) as a model [artifact](../../artifacts/intro.md) to a W&B run and link it to the [W&B Model Registry](../../model_registry/intro.md). 

When you link a model artifact to the registry, this creates a new version of that registered model. The new version is a pointer to the artifact version that exists in that project.

The following code snippet is a continuation of the code snippet in [Download and use a logged model](#download-and-use-a-logged-model). The code snippet uses the 

The proceeding code snippet shows how to link a model with the `link_model` API. The code snippet uses the `downloaded_model_path` variable defined in the [Download and use a logged model](#download-and-use-a-logged-model) section to provide the path of the model.

```python
run.link_model(path=downloaded_model_path,
                 registered_model_name="Industrial ViT",
                 linked_model_name=f"model_vit-{wandb.run.id}",
                 aliases=["staging", "QA"])

run.finish()
```