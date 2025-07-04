---
menu:
  default:
    identifier: create_collection
    parent: registry
title: Create a collection
weight: 4
---


A *collection* is a set of linked artifact versions within a registry. Each collection represents a distinct task or use case. 

For example, within the core Dataset registry you might have multiple collections. Each collection contains a different dataset such as MNIST, CIFAR-10, or ImageNet.

As another example, you might have a registry called "chatbot" that contains a collection for model artifacts, another collection for dataset artifacts, and another collection for fine-tuned model artifacts.

How you organize a registry and their collections is up to you.

{{% alert %}}
If you are familiar with W&B Model Registry, you might aware of registered models. Registered models in the Model Registry are now referred to as collections in the W&B Registry.
{{% /alert %}}

## Collection types

Each collection accepts one, and only one, *type* of artifact. The type you specify restricts what sort of artifacts you, and other members of your organization, can link to that collection.

{{% alert %}}
You can think of artifact types similar to data types in programming languages such as Python. In this analogy, a collection can store strings, integers, or floats but not a mix of these data types.
{{% /alert %}}

For example, suppose you create a collection that accepts "dataset" artifact types. This means that you can only link future artifact versions that have the type "dataset" to this collection. Similarly, you can only link artifacts of type "model" to a collection that accepts only model artifact types.

{{% alert %}}
You specify an artifact's type when you create that artifact object. Note the `type` field in `wandb.Artifact()`:

```python
import wandb

# Initialize a run
run = wandb.init(
  entity = "<team_entity>",
  project = "<project>"
  )

# Create an artifact object
artifact = wandb.Artifact(
    name="<artifact_name>", 
    type="<artifact_type>"
    )
```
{{% /alert %}}
 

When you create a collection, you can select from a list of predefined artifact types. The artifact types available to you depend on the registry that the collection belongs to. .

Before you link an artifact to a collection or create a new collection, [investigate the types of artifacts that collection accepts]({{< relref "#check-the-types-of-artifact-that-a-collection-accepts" >}}).

### Check the types of artifact that a collection accepts

Before you link to a collection, inspect the artifact type that the collection accepts. You can inspect the artifact types that collection accepts programmatically with the W&B Python SDK or interactively with the W&B App

{{% alert %}}
An error message appears if you try to create link an artifact to a collection that does not accept that artifact type.
{{% /alert %}}

{{< tabpane text=true >}}
  {{% tab header="W&B App" %}}
You can find the accepted artifact types on the registry card on the homepage or within a registry's settings page.

For both methods, first navigate to your W&B Registry App.

Within the homepage of the Registry App, you can view the accepted artifact types by scrolling to the registry card of that registry. The gray horizontal ovals within the registry card lists the artifact types that registry accepts.

{{< img src="/images/registry/artifact_types_model_card.png" alt="Artifact types selection" >}}

For example, the preceding image shows multiple registry cards on the Registry App homepage. Within the **Model** registry card, you can see two artifact types: **model** and **model-new**. 


To view accepted artifact types within a registry's settings page:

1. Click on the registry card you want to view the settings for.
2. Click on the gear icon in the upper right corner.
3. Scroll to the **Accepted artifact types** field.   
  {{% /tab %}}
  {{% tab header="Python SDK (Beta)" %}}
Programmatically view the artifact types that a registry accepts with the W&B Python SDK:

```python
import wandb

registry_name = "<registry_name>"
artifact_types = wandb.Api().project(name=f"wandb-registry-{registry_name}").artifact_types()
print(artifact_type.name for artifact_type in artifact_types)
```

{{% alert %}}
Note that you do not initialize a run with the proceeding code snippet. This is because it is unnecessary to create a run if you are only querying the W&B API and not tracking an experiment, artifact and so on.
{{% /alert %}}  
  {{% /tab %}}
{{< /tabpane >}}



Once you know what type of artifact a collection accepts, you can [create a collection]({{< relref "#create-a-collection" >}}).


## Create a collection

Interactively or programmatically create a collection within a registry. You can not change the type of artifact that a collection accepts after you create it.

### Programmatically create a collection

Use the `wandb.init.link_artifact()` method to link an artifact to a collection. Specify both the collection and the registry to the `target_path` field as a path that takes the form of:

```python
f"wandb-registry-{registry_name}/{collection_name}"
```

Where `registry_name` is the name of the registry and `collection_name` is the name of the collection. Ensure to append the prefix `wandb-registry-` to the registry name.

{{% alert %}}
W&B automatically creates a collection for you if you try to link an artifact to a collection that does not exist. If you specify a collection that does exists, W&B links the artifact to the existing collection.
{{% /alert %}}

The proceeding code snippet shows how to programmatically create a collection. Ensure to replace other the values enclosed in `<>` with your own:

```python
import wandb

# Initialize a run
run = wandb.init(entity = "<team_entity>", project = "<project>")

# Create an artifact object
artifact = wandb.Artifact(
  name = "<artifact_name>",
  type = "<artifact_type>"
  )

registry_name = "<registry_name>"
collection_name = "<collection_name>"
target_path = f"wandb-registry-{registry_name}/{collection_name}"

# Link the artifact to a collection
run.link_artifact(artifact = artifact, target_path = target_path)

run.finish()
```

### Interactively create a collection

The following steps describe how to create a collection within a registry using the W&B Registry App UI:

1. Navigate to the **Registry** App in the W&B App UI.
2. Select a registry.
3. Click on the **Create collection** button in the upper right hand corner.
4. Provide a name for your collection in the **Name** field. 
5. Select a type from the **Type** dropdown. Or, if the registry enables custom artifact types, provide one or more artifact types that this collection accepts.
6. Optionally provide a description of your collection in the **Description** field.
7. Optionally add one or more tags in the **Tags** field. 
8. Click **Link version**.
9. From the **Project** dropdown, select the project where your artifact is stored.
10. From the **Artifact** collection dropdown, select your artifact.
11. From the **Version** dropdown, select the artifact version you want to link to your collection.
12. Click on the **Create collection** button.

{{< img src="/images/registry/create_collection.gif" alt="Create a new collection" >}}