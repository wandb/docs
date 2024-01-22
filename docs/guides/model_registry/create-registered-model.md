---
description: ''
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Create a registered model

Create a [registered model](./model-management-concepts.md#registered-model) to hold all the candidate models for your modeling tasks. You can create a registered model interactively within the Model Registry or Artifact browser in the W&B App UI. 


## Interactively create a registered model
Interactively create a registered model within the [Model Registry App](https://wandb.ai/registry/model).


1. Navigate to the Model Registry App at [wandb.ai/registry/model](https://wandb.ai/registry/model).
![](/images/models/create_registered_model_1.png)
2. Click the **New registered model** button located in the top right of the Model Registry page.
![](/images/models/create_registered_model_model_reg_app.png)
3. From the panel that appears, select the entity you want the registered model to belong to from the **Owning Entity** dropdown.
![](/images/models/create_registered_model_3.png)
4. Provide a name for your model in the **Name** field. 
5. From the **Type** dropdown, select the type of artifacts to link to the registered model.
6. (Optional) Add a description about your model in the **Description** field. 
7. (Optional) Within the **Tags** field, add one or more tags. 
8. Click **Register model**.


:::tip
Manual linking a model to the model registry is useful for one-off models. However, it is often useful to [programmatically link model versions to the model registry](#programmatically-link-a-model).

For example, suppose you have a nightly job. It is tedious to manually link a model created each night. Instead, you could create a script that evaluates the model, and if the model improves in performance, link that model to the model registry with the W&B Python SDK.
:::

## Programmatically link a model
Programmatically link a model with the W&B Python SDK. W&B automatically creates a registered model for you if you try to link a model to the model registry that doesn't exist.

For example, suppose you have an existing registered model named "Fine-Tuned-Review-Autocompletion" in your Model Registry (see example [here](https://wandb.ai/reviewco/registry/model?selectionPath=reviewco%2Fmodel-registry%2FFinetuned-Review-Autocompletion&view=all-models)). And suppose that a few model versions are already linked to it: v0, v1, v2. If you programmatically link a model called `registered-model-name="Fine-Tuned-Review-Autocompletion"`, the new model is linked to this existing registered model as v3. If no registered model with this name exists, a new one registered model is created and the new model is linked as v0. 

Use the [`link_model`](../../ref/python/run.md#link_model) method to log model files to a W&B run and link it to the [W&B Model Registry](./intro.md).  

Ensure to replace other the values enclosed in `<>` with your own:

```python
import wandb

run = wandb.init(entity="<entity>", project="<project>")
run.link_model(path="<path-to-model>", registered_model_name="<registered-model-name>")
run.finish()
```

W&B creates a registered model for you if the name you specify for `registered-model-name` does not already exist. 

See [`link_model`](../../ref/python/run.md#link_model) in the API Reference guide for more information on optional parameters.
