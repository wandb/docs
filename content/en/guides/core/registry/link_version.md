---
menu:
  default:
    identifier: link_version
    parent: registry
title: Link an artifact version to a collection
weight: 5
---

Link an artifact version to a collection to make it available to other members in your organization. Linking an artifact to a collection brings that artifact version from a private, project-level scope, to a shared organization level scope.

You can [link an artifact version to a collection]({{< relref "/guides/core/registry/link_version.md#link-an-artifact-version-to-a-collection" >}}) programmatically with the W&B Python SDK or interactively with the W&B App.

When you link an artifact version to a collection, W&B creates a linked version of that artifact in the collection. The linked version points to the source artifact version that is logged to a run within a project. You can view the linked version in the collection and the source version in the project where it was logged. 

<!-- {{% alert %}}
The term "type" refers to the artifact object's type. When you create an artifact object ([`wandb.Artifact`]({{< relref "/ref/python/experiments/artifact.md" >}})), or log an artifact ([`wandb.init.log_artifact`]({{< relref "/ref/python/experiments/run.md#log_artifact" >}})), you specify a type for the `type` parameter. 
{{% /alert %}} -->

{{% alert %}}
Watch a [video demonstrating linking a version](https://www.youtube.com/watch?v=2i_n1ExgO0A) (8 min).
{{% /alert %}}

## Link an artifact to a collection 

Based on your use case, follow the instructions described in the tabs below to link an artifact version.

{{% alert %}}
Before you start, check the following:
* The types of artifacts that collection permits. For more information about collection types, see "Collection types" within [Create a collection]({{< relref "./create_collection.md" >}}).
* The registry that the collection belongs to already exists. To check that the registry exists, navigate to the [Registry App and search for]({{< relref "/guides/core/registry/search_registry" >}}) the name of the registry.
{{% /alert %}}



<!-- {{% alert %}}
If an artifact version logs metrics (such as by using `run.log_artifact()`), you can view metrics for that version from its details page, and you can compare metrics across artifact versions from the artifact's page. Refer to [View linked artifacts in a registry]({{< relref "#view-linked-artifacts-in-a-registry" >}}).
{{% /alert %}} -->



{{< tabpane text=true >}}
  {{% tab header="Python SDK" %}}


Programmatically link an artifact version to a collection with [`wandb.init.Run.link_artifact()`]({{< relref "/ref/python/experiments/run.md#link_artifact" >}}) or [`wandb.Artifact.link()`]({{< relref "/ref/python/experiments/artifact.md#method-artifactlink" >}}). Use `wandb.init.Run.link_artifact()` if you are linking an artifact version [within the context of a run](#link-an-artifact-version-within-the-context-of-a-run). Use `wandb.Artifact.link()` if you are linking an artifact version [outside the context of a run](#link-an-artifact-version-outside-the-context-of-a-run).

{{% alert %}}
`wandb.Artifact.link()` does not require you to initialize a run with `wandb.init()`. `wandb.init.Run.link_artifact()` requires you to initialize a run with `wandb.init()`.
{{% /alert %}}

For both approaches, specify the name of the artifact (`wandb.Artifact(type="<name>"`), the type  of artifact (`wandb.Artifact(type="<type>"`), and the `target_path` of the collection and registry you want to link the artifact version to.
<!-- 
For both methods, specify the path of the collection and registry you want to link the artifact version to with the `target_path` parameter.  -->

The target path consists of the prefix `"wandb-registry"`, the name of the registry, and the name of the collection separated by a forward slashes:

```text
wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}
```
<br>

<!-- 
{{% alert %}}
If you want to link an artifact version to the Model registry or the Dataset registry, set the artifact type to `"model"` or `"dataset"`, respectively.
{{% /alert %}} -->

<br>

### Link an artifact version within the context of a run

When you use `wandb.init.Run.link_artifact()`, you need to initialize a run with `wandb.init()`. This means that a run is created in your W&B project.

Copy and paste the code snippet below. Replace values enclosed in `<>` with your own:

```python
import wandb

entity = "<team_entity>"          # Your team entity
project = "<project_name>"        # The name of the project that contains your artifact

# Initialize a run
with wandb.init(entity = entity, project = project) as run:

  # Create an artifact object
  # The type parameter specifies both the type of the 
  # artifact object and the collection type
  artifact = wandb.Artifact(name = "<name>", type = "<type>")

  # Add the file to the artifact object. 
  # Specify the path to the file on your local machine.
  artifact.add_file(local_path = "<local_path_to_artifact>")

  # Specify the collection and registry to link the artifact to
  REGISTRY_NAME = "<registry_name>"  
  COLLECTION_NAME = "<collection_name>"
  target_path=f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"

  # Link the artifact to the collection
  run.link_artifact(artifact = artifact, target_path = target_path)
```

<br>

### Link an artifact version outside the context of a run

When you use `wandb.Artifact.link()`, you do not need to initialize a run with `wandb.init()`. This means that a run is not created in your W&B project. The artifact version is linked to the collection without being associated with a run. 

Copy and paste the code snippet below. Replace values enclosed in `<>` with your own:

```python
import wandb

# Create an artifact object
# The type parameter specifies both the type of the 
# artifact object and the collection type
artifact = wandb.Artifact(name = "<name>", type = "<type>")

# Add the file to the artifact object. 
# Specify the path to the file on your local machine.
artifact.add_file(local_path = "<local_path_to_artifact>")

# Save the artifact
artifact.save()  

# Specify the collection and registry to link the artifact to
REGISTRY_NAME = "<registry_name>"  
COLLECTION_NAME = "<collection_name>"
target_path=f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"

# Link the artifact to the collection
artifact.link(target_path = target_path)
```


  {{% /tab %}}
  {{% tab header="Registry App" %}}
1. Navigate to the Registry App.
    {{< img src="/images/registry/navigate_to_registry_app.png" alt="Registry App navigation" >}}
2. Hover your mouse next to the name of the collection you want to link an artifact version to.
3. Select the meatball menu icon (three horizontal dots) next to  **View details**.
4. From the dropdown, select **Link new version**.
5. From the sidebar that appears, select the name of a team from the **Team** dropdown.
5. From the **Project** dropdown, select the name of the project that contains your artifact. 
6. From the **Artifact** dropdown, select the name of the artifact. 
7. From the **Version** dropdown, select the artifact version you want to link to the collection.

<!-- TO DO insert gif -->  
  {{% /tab %}}
  {{% tab header="Artifact browser" %}}
1. Navigate to your project's artifact browser on the W&B App at: `https://wandb.ai/<entity>/<project>/artifacts`
2. Select the Artifacts icon on the left sidebar.
3. Click on the artifact version you want to link to your registry.
4. Within the **Version overview** section, click the **Link to registry** button.
5. From the modal that appears on the right of the screen, select an artifact from the **Select a register model** menu dropdown. 
6. Click **Next step**.
7. (Optional) Select an alias from the **Aliases** dropdown. 
8. Click **Link to registry**. 

<!-- Update this gif -->
<!-- {{< img src="/images/models/manual_linking.gif" alt="" >}} -->  
  {{% /tab %}}
{{< /tabpane >}}



<!-- {{% alert title="Linked vs source artifact versions" %}}
* Source version: the artifact version inside a team's project that is logged to a [run]({{< relref "/guides/models/track/runs/" >}}).
* Linked version: the artifact version that is published to the registry. This is a pointer to the source artifact, and is the exact same artifact version, just made available in the scope of the registry.
{{% /alert %}} -->


Once an artifact is linked, you can [view a linked artifact's metadata, version data, usage, lineage information]({{< relref "/guides/core/registry/link_version.md#view-linked-artifacts-in-a-registry" >}}) and more in the Registry App.

## View linked artifacts in a registry

View information about linked artifacts such as metadata, lineage, and usage information in the Registry App.

1. Navigate to the Registry App.
2. Select the name of the registry that you linked the artifact to.
3. Select the name of the collection.
4. If the collection's artifacts log metrics, compare metrics across versions by clicking **Show metrics**.
4. From the list of artifact versions, select the version you want to access. Version numbers are incrementally assigned to each linked artifact version starting with `v0`.
5. To view details about an artifact version, click the version. From the tabs in this page, you can view that version's metadata (including logged metrics), lineage, and usage information.

Make note of the **Full Name** field within the **Version** tab. The full name of a linked artifact consists of the registry, collection name, and the alias or index of the artifact version.

```text title="Full name of a linked artifact"
wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{INTEGER}
```

You need the full name of a linked artifact to access the artifact version programmatically.

## Troubleshooting 

Below are some common things to double check if you are not able to link an artifact. 

### Logging artifacts from a personal account

Artifacts logged to W&B with a personal entity can not be linked to the registry. Make sure that you log artifacts using a team entity within your organization. Only artifacts logged within an organization's team can be linked to the organization's registry. 


{{% alert title="" %}}
Ensure that you log an artifact with a team entity if you want to link that artifact to a registry.
{{% /alert %}}


#### Find your team entity

W&B uses the name of your team as the team's entity. For example, if your team is called **team-awesome**, your team entity is `team-awesome`.

You can confirm the name of your team by:

1. Navigate to your team's W&B profile page.
2. Copy the site's URL. It has the form of `https://wandb.ai/<team>`. Where `<team>` is the both the name of your team and the team's entity.

#### Log from a team entity
1. Specify the team as the entity when you initialize a run with [`wandb.init()`]({{< relref "/ref/python/functions/init.md" >}}). If you do not specify the `entity` when you initialize a run, the run uses your default entity which may or may not be your team entity.

  ```python 
  import wandb   

  run = wandb.init(
    entity='<team_entity>', 
    project='<project_name>'
    )
  ```

2. Log the artifact to the run either with run.log_artifact or by creating an Artifact object and then adding files to it with:

    ```python
    artifact = wandb.Artifact(name="<artifact_name>", type="<type>")
    ```
    To log artifacts, see [Construct artifacts]({{< relref "/guides/core/artifacts/construct-an-artifact.md" >}}).
3. If an artifact is logged to your personal entity, you will need to re-log it to an entity within your organization.

### Confirm the path of a registry in the W&B App UI

There are two ways to confirm the path of a registry with the UI: create an empty collection and view the collection details or copy and paste the autogenerated code on the collection's homepage.

#### Copy and paste autogenerated code

1. Navigate to the Registry app at https://wandb.ai/registry/.
2. Click the registry you want to link an artifact to.
3. At the top of the page, you will see an autogenerated code block. 
4. Copy and paste this into your code, ensure to replace the last part of the path with the name of your collection.

{{< img src="/images/registry/get_autogenerated_code.gif" alt="Auto-generated code snippet" >}}

#### Create an empty collection

1. Navigate to the Registry app at https://wandb.ai/registry/.
2. Click the registry you want to link an artifact to.
4. Click on the empty collection. If an empty collection does not exist, create a new collection.
5. Within the code snippet that appears, identify the `target_path` field within `.link_artifact()`.
6. (Optional) Delete the collection.

{{< img src="/images/registry/check_empty_collection.gif" alt="Create an empty collection" >}}

For example, after completing the steps outlined, you find the code block with the `target_path` parameter:

```python
target_path = 
      "smle-registries-bug-bash/wandb-registry-Golden Datasets/raw_images"
```

Breaking this down into its components, you can see what you will need to use to create the path to link your artifact programmatically:

```python
ORG_ENTITY_NAME = "smle-registries-bug-bash"
REGISTRY_NAME = "Golden Datasets"
COLLECTION_NAME = "raw_images"
```

{{% alert %}}
Ensure that you replace the name of the collection from the temporary collection with the name of the collection that you want to link your artifact to.
{{% /alert %}}