---
description: ''
displayed_sidebar: default
---

# Log models to an experiment

The following page describes how to log a model to your project. 


## Log a single model

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

:::tip
If you are familiar with W&B Artifacts, you will note that the process to log a model is nearly identical to how you would normally create an artifact. 

With the exception that, when you create an artifact, you must specify that the `type` of the artifact is set to `"model"`. In other words:

```python
wandb.Artifact(name="<artifact-name>", type="model")
```
:::    

## Log multiple models

<!-- To do  -->
TO DO



## Organize models with tags
Use tags to organize registered models into categories and to search over those categories. 

1. Navigate to the W&B Model Registry app at [https://wandb.ai/registry/model](https://wandb.ai/registry/model).
2. Select a registered model.
2. Go to the **Model card** section.
3. Click the plus button (**+**) next to **Tags**.
4. Search or create a new tag.
