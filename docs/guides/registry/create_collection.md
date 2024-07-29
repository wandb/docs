---
displayed_sidebar: default
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Create a collection

Create a collection within a registry to organize your artifacts. A *collection* is a set of linked artifact versions in a registry. Each collection represents a distinct task or use case and serves as a container for a curated selection of artifact versions related to that task. 


The proceeding diagram shows the hierarchical relationship between a registry, collections, and versions.

![](/images/registry/registry_diagram_homepage.png)



:::tip
If you are familiar with W&B Model Registry, you might aware of "registered models". In W&B Registry, registered models are renamed to "collections". The way you [create a registered model in the Model Registry](../model_registry/create-registered-model.md) is nearly the same for creating a collection in the W&B Registry. The main difference being that a collection does not belong to an entity like registered models.
:::

## Collection types

When you create a collection, you must select the type of artifacts that you can link to that collection. Each collection accepts one, and only one, type of artifact.  The type of artifact that a collection can have is determined by the registry that that collection is in.

:::tip
You specify the type of an artifact when you create that artifact. Note the `type` field in `wandb.Artifact()`:

```python
import wandb

# Initialize a run
run = wandb.init(entity="<team_entity>", project="<project>")

# Create an artifact object
artifact = wandb.Artifact(
    name="<artifact_name>", 
    type="<artifact_type>"
    )
```
:::

For example, suppose you create a collection that accepts "dataset" artifacts types. This means that only of artifacts of type "dataset" can be linked to this collection.




### Check the types of artifact that a collection accepts

Before you create and link to a collection, check the artifact type that the collection accepts:

<Tabs
  defaultValue="ui"
  values={[
    {label: 'W&B App', value: 'ui'},
    {label: 'Python SDK (Beta)', value: 'programmatically'},
  ]}>
  <TabItem value="ui">

Check the artifact types that a collection accepts on the registry cards on the homepage or within a registry's settings page. For both methods, you must first navigate to your W&B Registry App at https://wandb.ai/registry/.


Within the homepage of the Registry App, you can view the accepted artifact types by scrolling to the registry card you are interested in. The gray horizontal ovals within the registry card lists the artifact types that that registry accepts.

For example, the proceeding image shows multiple registry cards on the Registry App homepage. Within the **Model** registry card, you can see two artifact types: **model** and **model-new**. 

![](/images/registry/artifact_types_model_card.png)


To view accepted artifact types within a registry's settings page:

1. Click on the registry card you want to view the settings for.
2. Click on the gear icon in the upper right corner.
3. Scroll to the **Accepted artifact types** field. 


  </TabItem>
  <TabItem value="programmatically">

Programmatically view the artifact types that a registry accepts with the W&B Python SDK:

```python
import wandb

registry_name = "<registryName>"
org_entity = "<org_entity>"
artifact_types = wandb.Api().project(name=f"wandb-registry-{registry_name}", entity=org_entity).artifact_types()
print(artifact_type.name for artifact_type in artifact_types)
```


  </TabItem>
</Tabs>




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
