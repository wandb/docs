---
description: ''
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Log models to an experiment

The following page describes how to log a model with the W&B Python SDK. 


There are two ways to track models in W&B:
1. Track a model with the `wandb.log` method
2. Create a [model artifact](./model-management-concepts.md#model-artifact).

:::tip
Question: Should I use wandb.log or create a model artifact?

:::


For the proceeding code snippets, replace values within `<>` with your own.


## Log a model with [INSERT]

Use the [`log_model`](../../../ref/python/run.md#log_model) to log a model artifact that contains content within a directory you specify. The [`log_model`](../../../ref/python/run.md#log_model) method also marks the resulting model artifact as an output of the W&B run. 

You can track a model's dependencies and the model's associations if you mark the model as the input or output of a W&B run. View the lineage of the model within the W&B App UI. See the [Explore and traverse artifact graphs](../../artifacts/explore-and-traverse-an-artifact-graph.md) page within the [Artifacts](../../artifacts/intro.md) chapter for more information.

Provide the path where your model file(s) are saved to the `path` parameter. The path can be a local file, directory, or [reference URI](../../artifacts/track-external-files.md#amazon-s3--gcs--azure-blob-storage-references) to an external bucket such as `s3://bucket/path`. 

Ensure to replace values enclosed in `<>` with your own.

```python
import wandb

# Initialize a W&B run
run = wandb.init(project="<your-project>", entity="<your-entity>")

# Log the model
run.log_model(path="<path-to-model>", name="<name>")
```

Optionally provide a name for the model artifact for the `name` parameter. If `name` is not specified, W&B will use the basename of the input path prepended with the run ID as the name. 






## Log a model with a model artifact

:::tip
If you are familiar with W&B Artifacts, you will note that the process to log a model is nearly identical to how you would normally create an artifact. 

With the exception that, when you log a model, you must specify the `type` of the artifact to `"*model"`. In other words:

```python
wandb.Artifact(name="<artifact-name>", type="model")
```
:::  



1. First, create an empty artifact object with `wandb.Artifact()`. Ensure that the string you pass to `type` parameter possess `"model"`.

    ```python 
    import wandb

    run = wandb.init(entity="<entity>", project="<project>")

    artifact = wandb.Artifact(name="<artifact-name>", type="model")
    ```

<!-- :::tip
Thinking of artifacts as a directory, you can think of the name you provide when you create an artifact object as the name of root directory.
::: -->


2. Next, add a model to the artifact. Specify the local path of your serialized model and a name to your model for the `local_path` and `name` parameters, respectively: 

    ```python 
    path = "path/to/model"
    model_name = "<model-name>" 

    artifact.add_file(local_path=path, name=model_name)
    ```
3. (Optional) Log training metrics 

    ```python
    run.log(data={"train_loss": 0.345, "val_loss": 0.456})
    ```
4.  Declare a dataset dependency if you want to associate training data with your logged model.

    There are three ways to declare a dataset dependency:

    <Tabs
    defaultValue="apple"
    values={[
        {label: 'Dataset stored in W&B', value: 'apple'},
        {label: 'Dataset stored on local filesystem', value: 'orange'},
        {label: 'Dataset stored on remote bucket', value: 'banana'},
    ]}>
    <TabItem value="apple">

    ```python
    dataset = wandb.use_artifact("[[entity/]project/]name:alias")
    ```

    </TabItem>
    <TabItem value="orange">

    ```python
    art = wandb.Artifact("dataset_name", "dataset")
    art.add_dir("path/to/data")  # or art.add_file("path/to/data.csv")
    dataset = wandb.use_artifact(art)
    ```

    </TabItem>
    <TabItem value="banana">

    ```python
    art = wandb.Artifact("dataset_name", "dataset")
    art.add_reference("s3://path/to/data")
    dataset = wandb.use_artifact(art)
    ```

    </TabItem>
    </Tabs>







