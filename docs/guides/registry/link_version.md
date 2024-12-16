---
title: Link an artifact version to a registry
---
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';


Link artifact versions to a collection to make them available to other members in your organization. 

When you link an artifact to a registry, this "publishes" that artifact to that registry. Any user that has access to that registry can access the linked artifact versions in the collection.

In other words, linking an artifact to a registry collection brings that artifact version from a private, project-level scope, to a shared organization level scope.

{{% alert %}}
The term "type" refers to the artifact object type. When you create an artifact object ([`wandb.Artifact`](../../ref/python/artifact.md)), or log an artifact ([`wandb.run.log_artifact`](../../ref/python/run.md#log_artifact)), you specify a type for the `type` parameter. 
<!-- If you are familiar with Python, you can think of artifact types in W&B as having similar functions as Python data types.  -->
{{% /alert %}}

## Link an artifact to collection

Link an artifact version to a collection interactively or programmatically. 

{{% alert %}}
Before you link an artifact to a registry, check the types of artifacts that collection permits. For more information about collection types, see "Collection types" within [Create a collection](./create_collection.md).
{{% /alert %}}

Based on your use case, follow the instructions described in the tabs below to link an artifact version.

<Tabs
  defaultValue="python_sdk"
  values={[
    {label: 'Python SDK', value: 'python_sdk'},
    {label: 'Registry App', value: 'registry_ui'},
    {label: 'Artifact browser', value: 'artifacts_ui'},
  ]}>
  <TabItem value="python_sdk">


Programmatically link an artifact version to a collection with [`wandb.init.Run.link_artifact()`](../../ref/python/run.md#link_artifact). 


:::info
Before you link an artifact to a collection, ensure that the registry that the collection belongs to already exists. To check that the registry exists, navigate to the Registry app on the W&B App UI and search for the name of the registry.
:::

Use the `target_path` parameter to specify the collection and registry you want to link the artifact version to. The target path consists of the prefix "wandb-registry", the name of the registry, and the name of the collection separated by a forward slashes:

```text
wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}
```

Copy and paste the code snippet below to link an artifact version to a collection within an existing registry. Replace values enclosed in `<>` with your own:

```python
import wandb

# Initialize a run
run = wandb.init(
  entity = "<team_entity>",
  project = "<project_name>"
)

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

:::info
If you want to link an artifact version to the Model registry or the Dataset registry, set the artifact type to `"model"` or `"dataset"`, respectively.
:::


  </TabItem>
  <TabItem value="registry_ui">

1. Navigate to the Registry App.
{{< img src="/images/registry/navigate_to_registry_app.png" alt="" >}}
2. Hover your mouse next to the name of the collection you want to link an artifact version to.
3. Select the meatball menu icon (three horizontal dots) next to  **View details**.
4. From the dropdown, select **Link new version**.
5. From the sidebar that appears, select the name of a team from the **Team** dropdown.
5. From the **Project** dropdown, select the name of the project that contains your artifact. 
6. From the **Artifact** dropdown, select the name of the artifact. 
7. From the **Version** dropdown, select the artifact version you want to link to the collection.

<!-- TO DO insert gif -->

  </TabItem>
  <TabItem value="artifacts_ui">

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

  </TabItem>
</Tabs>



<!-- {{% alert title="Linked vs source artifact versions" %}}
* Source version: the artifact version inside a team's project that is logged to a [run](../runs/intro.md).
* Linked version: the artifact version that is published to the registry. This is a pointer to the source artifact, and is the exact same artifact version, just made available in the scope of the registry.
{{% /alert %}}
 -->

View a linked artifact's metadata, version data, usage, lineage information and more in the Registry App.

## View linked artifacts in a registry

View information about linked artifacts such as metadata, lineage, and usage information in the Registry App.

1. Navigate to the Registry App.
2. Select the name of the registry that you linked the artifact to.
3. Select the name of the collection.
4. From the list of artifact versions, select the version you want to access. Version numbers are incrementally assigned to each linked artifact version starting with `v0`.

Once you select an artifact version, you can view that version's metadata, lineage, and usage information.

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
Ensure that you you log an artifact with a team entity if you want to link that artifact to a registry.
{{% /alert %}}


#### Find your team entity

W&B uses the name of your team as the team's entity. For example, if your team is called "team-awesome", your team entity is `team-awesome`.

You can confirm the name of your team by:

1. Navigate to your team's W&B profile page.
2. Copy the site's URL. It has the form of `https://wandb.ai/<team>`. Where `<team>` is the both the name of your team and the team's entity.

#### Log from a team entity
1. Specify the team as the entity when you initialize a run with [`wandb.init()`](/ref/python/init). If you do not specify the `entity` when you initialize a run, the run uses your default entity which may or may not be your team entity. 
  ```python 
  import wandb   

  run = wandb.init(
    entity='<team_entity>', 
    project='<project_name>'
    )
  ```
2. Log the artifact to the run either with run.log_artifact or by creating an Artifact object and then adding files to it with  :

    ```python
    artifact = wandb.Artifact(name="<artifact_name>", type="<type>")
    run.log_artifact(artifact)
    ```
    For more information on how to log artifacts, see [Construct artifacts](../artifacts/construct-an-artifact.md).
3. If an artifact is logged to your personal entity, you will need to re-log it to an entity within your organization.

### Confirm the path of a registry in the W&B App UI

There are two ways to confirm the path of a registry with the UI: create an empty collection and view the collection details or copy and paste the autogenerated code on the collection's home page.

#### Copy and paste autogenerated code

1. Navigate to the Registry app at https://wandb.ai/registry/.
2. Click the registry you want to link an artifact to.
3. At the top of the page, you will see an autogenerated code block. 
4. Copy and paste this into your code, ensure to replace the last part of the path with the name of your collection.

{{< img src="/images/registry/get_autogenerated_code.gif" alt="" >}}

#### Create an empty collection

1. Navigate to the Registry app at https://wandb.ai/registry/.
2. Click the registry you want to link an artifact to.
4. Click on the empty collection. If an empty collection does not exist, create a new collection.
5. Within the code snippet that appears, identify the `target_path` field within `.link_artifact()`.
6. (Optional) Delete the collection.

{{< img src="/images/registry/check_empty_collection.gif" alt="" >}}

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

