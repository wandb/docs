---
description: ''
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Create a registered model

Create a [registered model](./model-management-concepts.md#registered-model) to hold all the candidate models for your modeling tasks. You can create a registered model interactively within the Model Registry or Artifact browser in the W&B App UI. 


## Interactively create a registered model
Interactively create a registered model within the [Model Registry App](https://wandb.ai/registry/model) or within your project's artifact browser.


<Tabs
  defaultValue="registry"
  values={[
    {label: 'Model Registry', value: 'registry'},
    {label: 'Artifact browser', value: 'browser'},
  ]}>
  <TabItem value="registry">

1. Navigate to the Model Registry App at [wandb.ai/registry/model](https://wandb.ai/registry/model).
![](/images/models/create_registered_model_1.png)
2. Click the **New registered model** button located in the top right of the Model Registry page.
![](/images/models/create_registered_model_model_reg_app.png)
3. A panel will appear. Select the entity you want the registered model to belong to from the **Owning Entity** dropdown.
![](/images/models/create_registered_model_3.png)
4. Provide a name for your model in the **Name** field. 
5. From the **Type** dropdown, select the type of artifacts to link to the registered model.
6. (Optional) Add a description about your model in the **Description** field. 
7. (Optional) Within the **Tags** field, add one or more tags. 
8. Click **Register model**.


  </TabItem>
  <TabItem value="browser">

1. Navigate to your project's artifact browser on the W&B App at: `https://wandb.ai/<entity>/<project>/artifacts`
2. Click the **+** icon on the bottom of the sidebar.
3. From the **Type** dropdown, select **model**.
3. From the **Style** dropdown, select **Registered model**.
4. Provide a name for your model registry in the **Name** field. W&B suggests that you enter a unique name that describes the use case for this model.
5. Select **Create**.

![](/images/models/artifact_browser.gif)

  </TabItem>
</Tabs>

:::tip
Manual linking a model to the model registry is useful for one-off models. However, it is often useful to programmatically link model versions to the model registry. For example, suppose you have a nightly job. It is tedious to manually link a model created each night.Instead, you could create a script that evaluate the model, and if the model improves in performance, link that model to the model registry with the W&B Python SDK.
:::

## Programmatically link a model
Programmatically link a model with the W&B Python SDK. W&B will automatically create a registered model for you if you try to link a model to the model registry that doesn't exist.


For example, suppose you have an existing registered model named "Fine-Tuned-Review-Autocompletion" in your Model Registry (see example [here](https://wandb.ai/reviewco/registry/model?selectionPath=reviewco%2Fmodel-registry%2FFinetuned-Review-Autocompletion&view=all-models)). And suppose that a few model versions are already linked to it: v0, v1, v2. If you programmatically link a model called `registered-model-name="Fine-Tuned-Review-Autocompletion"`, the new model will be linked to this existing registered model as v3. If no registered model with this name exists, a new one will be created and the new model will be linked as v0. 

Depending on your context and use case, use one of the APIs listed below.


<Tabs
  defaultValue="within"
  values={[
    {label: 'Within a run', value: 'within'},
    {label: 'Outside of a run', value: 'public'},
  ]}>
  <TabItem value="within">

Use the [`link_model`](../../ref/python/run.md#link_model) method to log model file(s) to a W&B run and link it to the [W&B Model Registry](./intro.md).  

Ensure to replace other the values enclosed in `<>` with your own:

```python
import wandb

run = wandb.init(entity="<your-entity>", project="<your-project>")
run.link_model(path="<path-to-model>", registered_model_name="<registered-model-name>")
run.finish()
```

W&B will create a registered model for you if the name you specify for `registered-model-name` does not already exist. 

See [`link_model`](../../ref/python/run.md#link_model) in the API Reference guide for more information on optional parameters.

  </TabItem>
    <TabItem value="public">

Use the W&B Public API and [W&B artifacts](../artifacts/intro.md) to log a model outside of a W&B run.

Ensure to replace other the values enclosed in `<>` with your own:

```python
import wandb

entity = "<entity>"
registered_model_name = "<registered-model-name>"
artifact_name = "<artifact-name:alias>"  

# Fetch the Model Version via API
artifact = wandb.Api().artifact(name=artifact_name)

# Link the model version to the Model Registry
target_path = f"{entity}/model-registry/{registered_model_name}"
artifact.link(target_path=target_path)
```

W&B will create a registered model for you if the name you specify for `registered-model-name` does not already exist. 

For more information about artifacts, see [LINK].

  </TabItem>
</Tabs> 