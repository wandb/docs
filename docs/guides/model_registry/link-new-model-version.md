---
description: ''
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Link a new model version

Link a model version to the registered model with the W&B App or programmatically with the Python SDK.

## Programmatically link a model


<Tabs
  defaultValue="within"
  values={[
    {label: 'Within a run', value: 'within'},
    {label: 'Outside of a run', value: 'public'},
  ]}>
  <TabItem value="within">

Link a model version within a W&B run with the `link_model` API.

Use the [`link_model`](../../ref/python/run.md#link_model) method to log model file(s) to a W&B run and link it to the [W&B Model Registry](./intro.md). If the `registered-model-name` matches the name of a registered model that already exists within the Model Registry, the model will be linked to that registered model. If no such registered model exists, a new one will be created and the model will be the first one linked. 

The proceeding code snippet shows how to link a model with the [`link_model`](../../ref/python/run.md#link_model) API. Ensure to replace other the values enclosed in `<>` with your own:

```python
import wandb

run = wandb.init(entity="<your-entity>", project="<your-project>")
run.link_model(path="<path-to-model>", registered_model_name="<registered-model-name>")
run.finish()
```

  </TabItem>
  <TabItem value="public">

Fetch Model Artifact from Public API:

```python
import wandb

artifact_name = "artifact-collection:alias"  # Name of artifact collection

# Fetch the Model Version via API
artifact = wandb.Api().artifact(name=artifact_name)

# Link the Model Version to the Model Collection
target_path = f"{entity}/model-registry/{registered_model_name}"
artifact.link(target_path=target_path)
```

  </TabItem>  
</Tabs>



## Interactively link a model


1. Navigate to your project's artifact browser on the W&B App at: `https://wandb.ai/<entity>/<project>/artifacts`
2. Select the Artifacts icon on the left sidebar.
3. Click on the model version you want to link to your registry.
4. Within the **Version overview** section, click the **Link to registry** button.
5. From the modal that appears on the right of the screen, select a registered model from the **Select a register model** menu dropdown. 
6. Click **Next step**.
7. (Optional) Select an alias from the **Aliases** dropdown. 
8. Click **Link to registry**. 

![](/images/models/manual_linking.gif)

## View the source of linked models

There are two ways to view the source of linked models: The artifact browser within the project that the model was logged to and the model registry.

After you link a [model version](./model-management-concepts.md#model-version), you will see a pointer that connects a specific model version in the model registry to the source model artifact (located within the project the model was logged to). The source model artifact will also have a pointer that points to the model registry.


<Tabs
  defaultValue="registry"
  values={[
    {label: 'Model Registry', value: 'registry'},
    {label: 'Artifact browser', value: 'browser'},
  ]}>
  <TabItem value="registry">

1. Navigate to your model registry at [wandb.ai/registry/model](https://wandb.ai/registry/model).
![](/images/models/create_registered_model_1.png)
2. Select **View details** next the name of your registered model.
3. Within the **Versions** section, select **View** next to the model version you want to investigate.
4. Click on the **Version** tab within the right panel.
5. Within the **Version overview** section you will see a row that contains a **Source Version** field. The **Source Version** field shows both the name of the model and the model's version.

For example, the following image shows that a `v0` [model version](./model-management-concepts.md#model-version) called `mnist_model`, was linked to the `MNIST-dev` model registry.

![](/images/models/view_linked_model_registry.png)

  </TabItem>
  <TabItem value="browser">

1. Navigate to your project's artifact browser on the W&B App at: `https://wandb.ai/<entity>/<project>/artifacts`
2. Select the Artifacts icon on the left sidebar.
3. Expand the **model** dropdown menu from the Artifacts panel.
4. Select the name and version of the model that is linked to the model registry.
5. Click on the **Version** tab within the right panel.
6. Within the **Version overview** section you will see a row that contains a **Linked To** field. The **Linked To** field shows both the name of the registered model and the version it was given (`registered-model-name:version`). 

For example, the following image shows that a `v0` [model version](./model-management-concepts.md#model-version) called `mnist_model`, was linked to the `MNIST-dev` model registry. 

![](/images/models/view_linked_model_artifacts_browser.png)


  </TabItem>
</Tabs>