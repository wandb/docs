---
description: ''
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Create a registered model

Create a registered model to hold all the candidate models for your modeling tasks.

You can create a registered model interactively within the Model Registry or Artifact browser in the W&B App UI. 

<Tabs
  defaultValue="registry"
  values={[
    {label: 'Model Registry', value: 'registry'},
    {label: 'Artifact browser', value: 'browser'},
  ]}>
  <TabItem value="registry">

1. Navigate to your model registry at [wandb.ai/registry/model](https://wandb.ai/registry/model).
![](/images/models/create_registered_model_1.png)
2. Click the **New registered model** button at the top of the Model Registry page.
![](/images/models/create_registered_model_3.png)
3. Select the entity the registered model will belong to from the **Owning Entity** dropdown.
4. Provide a name for your model in the **Model Name** field. 


  </TabItem>
  <TabItem value="browser">

1. Navigate to your project's artifact browser on the W&B App at: `https://wandb.ai/<entity>/<project>/artifacts`
2. Click the **+** icon on the bottom of the sidebar.
3. From the **Type** dropdown, select **model**.
3. From the **Style** dropdown, select **Collection**.
4. Provide a name for your model registry in the **Name** field. W&B suggests that you enter a unique name that describes the use case for this model.

![](/images/models/browser.gif)

  </TabItem>
</Tabs>

:::tip
You can also create a registered model with the W&B Python SDK. To do so, you can skip creating a registered model with the W&B App UI and directly link a model to a non-existing registered model. W&B will automatically create a registered model if you try to link a model to a registry that does not already exist. For more information on how to link a model, see the [Link a new model version](./link-new-model-version.md) page.
:::

<!-- 
## Programmatically register a model

<Tabs
  defaultValue="public"
  values={[
    {label: 'Outside of a run', value: 'public'},
    {label: 'Within a run', value: 'within'},
    {label: 'Logged by current run', value: 'current'},
  ]}>
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
  <TabItem value="within">

Model Artifact is "used" by the current Run

```python
import wandb

entity = "<entity>"
project = "<project>"  # Project where your artifact exists
artifact_name = "artifact-collection:alias"  # Name of artifact collection
registered_model_name = "<model-name>"  # Name for your registered model

# Initialize a W&B run to start tracking
run = wandb.init(entity=entity, project=project)

# Obtain a reference to a Model Version
artifact = run.use_artifact(artifact_or_name=artifact_name)

# Link the Model Version to the Model Collection
target_path = f"{entity}/model-registry/{registered_model_name}"
artifact.link(target_path=target_path)
```
  </TabItem>
  <TabItem value="current">

Model Artifact is logged by the current Run

```python
import wandb

entity = "<entity>"
project = "<project>"  # Project where your artifact exists
artifact_name = "artifact-collection"  # Name of artifact collection
registered_model_name = "<model-name>"  # Name for your registered model

# Initialize a W&B run to start tracking
run = wandb.init(entity=entity, project=project)

# Create an Model Version
artifact = wandb.Artifact(name=artifact_name, type="model")

# Log the Model Version
run.log_artifact(artifact)

# Link the Model Version to the Collection
target_path = f"{entity}/model-registry/{registered_model_name}"
run.link_artifact(artifact=artifact, target_path=target_path, aliases=["Best"])
```

  </TabItem>
</Tabs> -->


