---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Create a collection

Create a collection within a registry to organize your artifacts. A *collection* is a set of linked artifact versions in a registry. Each collection represents a distinct task or use case and serves as a container for a curated selection of artifact versions related to that task. 


The proceeding diagram shows the hierarchical relationship between a registry, collections, and versions.

![](/images/registry/registry_diagram_homepage.png)


<!-- What the type field is and how do you choose this? We need to convey in there that
 you will only be able to link artifacts that match this collection type, and 
 the collection type has to be one of the allowed ones for this registry -->




:::tip
If you are familiar with W&B Model Registry, you might aware of "registered models". In W&B Registry, registered models are renamed to "collections". The way you [create a registered model in the Model Registry](../model_registry/create-registered-model.md) is nearly the same for creating a collection in the W&B Registry. The main difference being that a collection does not belong to an entity like registered models.
:::

## Collection type

Each collection consists of one or more collection types. 

The types of artifacts you can link to a collection is determined based on the  both the collection and the registry. This means that you can only link an artifact to a collection if the artifact has the same type that the registry permits.


When a custom registry is created, 


## Programmatically create a collection
Programmatically create a collection with the W&B Python SDK. W&B automatically creates a collection with the name you specify in the target path if you try to link an artifact to a collection that does not exist. The target path consists of the entity of the organization, the prefix "wandb-registry-", the name of the registry, and the name of the collection:

```python
f"{org_entity}/wandb-registry-{registry_name}/{collection_name}"
```

The proceeding code snippet shows how to programmatically create a collection. Ensure to replace other the values enclosed in `<>` with your own:

```python
import wandb

# Initialize a run
run = wandb.init(entity="<team_entity>", project="<project>")

# Create an artifact object
artifact = wandb.Artifact(name="<artifact_name>", type="<artifact_type>")

org_entity = "<organization_entity>"
registry_name = "<registry_name>"
collection_name = "<collection_name>"
target_path = f"{org_entity}/wandb-registry-{registry_name}/{collection_name}"

# Link the artifact to a collection
run.link_artifact(artifact = artifact, target_path = target_path)

run.finish()
```


## Interactively create a collection

The following steps describe how to create a collection within a registry using the W&B Registry App UI:

1. Navigate to the **Registry** App in the W&B App UI.
2. Select a registry.
3. Click on the **Create collection** button in the upper right hand corner.
4. Provide a name for your collection in the **Name** field. 
5. Select a type from the **Type** dropdown. Or, if the registry enables custom artifact types, provide one or more artifact types that this collection accepts.
:::info
An artifact type can not be removed from a registry once it is added and saved in the registry's settings.
:::
5. Optionally provide a description of your collection in the **Description** field.
6. Optionally add one or more tags in the **Tags** field. 
7. Click **Link version**.
8. From the **Project** dropdown, select the project where your artifact is stored.
9. From the **Artifact** collection dropdown, select your artifact.
10. From the **Version** dropdown, select the artifact version you want to link to your collection.
11. Click on the **Create collection** button.

![](/images/registry/create_collection.gif)





For example, the proceeding image shows a registry named "Fine-tuned models". Within the "Fine-tuned models" registry there are is a collection called "MNIST".

![](/images/registry/what_is_collection.png)