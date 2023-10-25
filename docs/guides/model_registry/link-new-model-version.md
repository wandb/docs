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
</Tabs>

## Interactively link a model


1. Navigate to the project that contain your logged models.
2. Select the Artifacts icon on the left sidebar.
3. Click the model version you want to link to your registry.
4. Within the **Version overview** section, click the **Link to registry** button.
5. From the modal that appears on the right of the screen, select a registered model from the **Select a register model** menu dropdown. 
6. Click **Next step**.
7. (Optional) Select an alias from the **Aliases** dropdown. 
8. Click **Link to registry**. 


## View linked models

After you link the model version, you will see hyperlinks that connect the version in the registered model to the source artifact. The artifact will also have hyperlinks that connect to the model version.

![](@site/static/images/models/train_log_model_version.png)
