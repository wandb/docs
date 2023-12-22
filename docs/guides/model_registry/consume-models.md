---
description: ''
displayed_sidebar: default
---

# Download a model version

Use the W&B Python SDK to download a model that you have linked to the Model Registry. This is particularly useful if you want to evaluate its performance, make predictions against a dataset, or use in a live production context. 

:::info
You are responsible for providing additional Python functions, API calls to reconstruct, deserialize your model into a form that you can work with. 

W&B suggests that you document information on how to load models into memory with model cards. For more information, see the [Document machine learning models](./create-model-cards.md) page. 
:::

The following code snippet shows how to download a model version with the W&B Python SDK. Replace values within `<>` with your own:

```python
import wandb

# Initialize a run
run = wandb.init(project="<your-project>", entity="<your-entity>")

# Access and download model. Returns path to downloaded artifact
downloaded_model_path = run.use_model(name="<your-model-name>")
```

Reference a model version with one of the formats listed:

* `latest` - which will fetch the most recently linked Version
* `v#` - using `v0`, `v1`, `v2`, ... you can fetch a specific version in the Registered Model
* `alias` - specify the custom alias that you and your team assigned to your model version


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

downloaded_model_path = run.use_model(name=f"{model_artifact_name}:{alias}")
```
</details>

See [`use_model`](../../ref/python/run.md#use_model) in the API Reference guide for more information on possible parameters and return type.
