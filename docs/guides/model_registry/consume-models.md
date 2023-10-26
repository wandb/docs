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

entity = "<entity>"
project = "<project>"  # Project where your artifact exists
job_type = "<>"  # (Optional)
registered_model_name = "<model-name>"  # Name for your registered model
alias = "<alias>"

artifact_name = f"{entity}/model-registry/{registered_model_name}:{alias}"

run = wandb.init(project=project, entity=entity, job_type=job_type)

artifact = run.use_artifact(artifact_or_name=artifact_name, type="model")
artifact_dir = artifact.download()
wandb.finish()
```

You can reference a version within a registered model using different alias strategies:

* `latest` - which will fetch the most recently linked Version
* `v#` - using `v0`, `v1`, `v2`, ... you can fetch a specific version in the Registered Model
* `production` - you can use any custom alias that you and your team have assigned


