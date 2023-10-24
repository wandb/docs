---
description: ''
displayed_sidebar: default
---

# Log a model to an experiment

The following page describes how to log a model to your project. 

:::tip
If you are familiar with W&B Artifacts, you will note that the process to log a model is nearly identical to how you would normally create an artifact. 

With the exception that, when you create an artifact, you must specify that the type of the artifact is set to `"model"`. In other words:

```python
wandb.Artifact(name="<artifact-name>", type="model")
```
:::


<!-- ## Log a single model -->

1. First, create an empty artifact object. Ensure to specify `"model"` as the type when you create the artifact.

    The following code snippet shows how to create an empty artifact. Replace the values within `<>` with your own:

    ```python 
    import wandb

    run = wandb.init(entity="<entity>", project="<project>")

    artifact = wandb.Artifact(name="<artifact-name>", type="model")
    ```

<!-- :::tip
Thinking of artifacts as a directory, you can think of the name you provide when you create an artifact object as the name of root directory.
::: -->


2. Next, add the model to the artifact. Specify the local path of your serialized model and a name to your model for the `local_path` and `name` parameters, respectively: 

    ```python 
    path = "path/to/model"
    model_name = "<model-name>"  # model.h5

    artifact.add_file(local_path=path, name=model_name)
    ```

<!-- Do I need this?: artifact.save() -->

<!-- ## Log multiple models 
In some use cases, you might want to log multiple versions of a model. For example, you might want log model checkpoints.

In such cases, W&B suggests that you use an alias. You can use any alias that you prefer. In particular, W&B suggests that you add a `"best"` alias to the model that outperforms all other model versions.


Suppose you have 

```python
import wandb

run = wandb.init(entity="<entity>", project="<project>")

artifact = wandb.Artifact(name="<artifact-name>", type="model")

path = "path/to/model"
model_name = "<model-name>"

artifact.add_file(local_path=path, name=model_name)

wandb.log_artifact(artifact, aliases="best")
``` -->