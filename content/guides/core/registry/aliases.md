---
title: Reference an artifact version with aliases
weight: 5
---

Reference a specific [artifact version]({{< relref "guides/core/artifacts/create-a-new-artifact-version" >}}) with one or more aliases. [W&B automatically assigns aliases]({{< relref "aliases#default-aliases" >}}) to each artifact you link with the same name. You can also [create one or more custom aliases]({{< relref "aliases#custom-aliases" >}}) to reference a specific artifact version.

Aliases appear as rectangles with the name of that alias in the rectangle in the Registry UI. If an [alias is protected]({{< relref "aliases#protected-aliases" >}}), it appears as a gray rectangle with a lock icon. Otherwise, the alias appears as an orange rectangle. Aliases are not shared across registries.

{{% alert title="When to use an alias versus using a tag" %}}
Use an alias to reference a specific artifact version. Each alias within a collection is unique. Only one artifact version can have a specific alias at a time.

Use tags to organize and group artifact versions or collections based on a common theme. Multiple artifact versions and collections can share the same tag.
{{% /alert %}}

When you add an alias to an artifact version, you can optionally start a [Registry automation]({{< relref  "/guides/core/automations/automation-events/#registry" >}}) to notify a Slack channel or trigger a webhook.

## Default aliases

W&B automatically assigns the following aliases to each artifact version you link with the same name:

* The `latest` alias to the most recent artifact version you link to a collection.
* A unique version number. W&B counts each artifact version (zero indexing) you link. W&B uses the count number to assign a unique version number to that artifact.

For example, if you link an artifact named `zoo_model` three times, W&B creates three aliases `v0`, `v1`, and `v2` respectively. `v2` also has the `latest` alias.

## Custom aliases

Create one or more custom aliases for a specific artifact versions based on your unique use case. For example:

- You might use aliases such as `dataset_version_v0`, `dataset_version_v1`, and `dataset_version_v2` to identify which dataset a model was trained on.
- You might use a `best_model` alias to keep track of the best performing artifact model version.

Any user with a [Member or Admin registry role]({{< relref "guides/core/registry/configure_registry/#registry-roles" >}}) on a registry can add or remove a custom alias from a linked artifact in that registry. If appropriate, use [protected aliases]({{< relref "aliases/#protected-aliases" >}}) to label and identify which artifact versions to protect from modification or deletion.

You can create a custom alias with the W&B Registry or the Python SDK. Based on your use case, click on a tab below that best fits your needs.

{{< tabpane text=true >}}
{{% tab header="W&B Registry" value="app" %}}

1. Navigate to the W&B Registry.
2. Click the **View details** button in a collection.
3. Within the **Versions** section, click the **View** button for a specific artifact version.
4. Click the **+** button to add one or more aliases next to the **Aliases** field.

{{% /tab %}}

{{% tab header="Python SDK" value="python" %}}
When you link an artifact version to a collection with the Python SDK you can optionally provide a list of one or more aliases as an argument to the `alias` parameter in [`link_artifact()`]({{< relref "/ref/python/run/#link_artifact" >}}). W&B creates an alias ([non protected alias]({{< relref "#custom-aliases" >}})) for you if the alias you provide does not already exist.

The following code snippet demonstrates how to link an artifact version to a collection and add aliases to that artifact version with the Python SDK. Replace values within `<>` with your own:

```python
import wandb

# Initialize a run
run = wandb.init(entity = "<team_entity>", project = "<project_name>")

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

# Link the artifact version to the collection
# Add one or more aliases to this artifact version
run.link_artifact(
    artifact = artifact, 
    target_path = target_path, 
    aliases = ["<alias_1>", "<alias_2>"]
    )
```
{{% /tab %}}
{{< /tabpane >}}

### Protected aliases
Use a [protected alias]({{< relref "aliases/#protected-aliases" >}}) to both label and identify artifact versions that should not be modified or deleted. For example, consider using a `production` protected alias to label and identify artifact versions that are in used in your organization's machine learning production pipeline.

Only [registry admins]({{< relref "/guides/core/registry/configure_registry/#registry-roles" >}}) can create protected aliases and add or remove protected aliases from an artifact version. Members and Viewers cannot unlink a protected version or delete a collection that contains a protected . See [Configure registry access]({{< relref "/guides/core/registry/configure_registry.md" >}}) for details.

Common protected aliases include:

- **Production**: The artifact version is ready for production use.
- **Staging**: The artifact version is ready for testing.

#### Create a protected alias

The following steps describe how to create a protected alias in the W&B Registry UI:

1. Navigate to the Registry App.
2. Select a registry.
3. Click the gear button on the top right of the page to view the registry's settings.
4. Within the **Protected Aliases** section, click the **+** button to add one or more protected aliases.

After creation, each protected alias appears as a gray rectangle with a lock icon in the **Protected Aliases** section.  

{{% alert %}}
Unlike custom aliases that are not protected, creating protected aliases is available exclusively in the W&B Registry UI and not programmatically with the Python SDK. To add a protected alias to an artifact version, you can use the W&B Registry UI or the Python SDK.
{{% /alert %}}

The following steps describe how to add a protected alias to an artifact version with the W&B Registry UI:

1. Navigate to the W&B Registry.
2. Click the **View details** button in a collection.
3. Within the **Versions** section, select the **View** button for a specific artifact version.
4. Click the **+** button to add one or more protected aliases next to the **Aliases** field.

After a protected alias is created, an admin can add it to an artifact version programmatically with the Python SDK. See the W&B Registry and Python SDK tabs in [Create a custom alias](#custom-aliases) section above for an example on how to add a protected alias to an artifact version.

## Find existing aliases
You can find existing aliases with the [global search bar in the W&B Registry]({{< relref "/guides/core/registry/search_registry/#search-for-registry-items" >}}). To find a protected alias:

1. Navigate to the W&B Registry App.
2. Specify the search term in the search bar at the top of the page. Press Enter to search.

Search results appear below the search bar if the term you specify matches an existing registry, collection name, artifact version tag, collection tag, or alias.

## Example

{{% alert %}}
The following code example is a continuation of [the W&B Registry Tutorial](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb_registry/zoo_wandb.ipynb). To use the following code, you must first [retrieve and process the Zoo dataset as described in the notebook](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb_registry/zoo_wandb.ipynb#scrollTo=87fecd29-8146-41e2-86fb-0bb4e3e3350a). Once you have the Zoo dataset, you can create an artifact version and add custom aliases to it.
{{% /alert %}}

The following code snippet shows how to create an artifact version and add custom aliases to it. The example uses the Zoo dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/111/zoo) and the `Model` collection in the `Zoo_Classifier_Models` registry. 

```python
import wandb

# Initialize a run
run = wandb.init(entity = "smle-reg-team-2", project = "zoo_experiment")

# Create an artifact object
# The type parameter specifies both the type of the 
# artifact object and the collection type
artifact = wandb.Artifact(name = "zoo_dataset", type = "dataset")

# Add the file to the artifact object. 
# Specify the path to the file on your local machine.
artifact.add_file(local_path="zoo_dataset.pt", name="zoo_dataset")
artifact.add_file(local_path="zoo_labels.pt", name="zoo_labels")

# Specify the collection and registry to link the artifact to
REGISTRY_NAME = "Model"
COLLECTION_NAME = "Zoo_Classifier_Models"
target_path=f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"

# Link the artifact version to the collection
# Add one or more aliases to this artifact version
run.link_artifact(
    artifact = artifact,
    target_path = target_path,
    aliases = ["production-us", "production-eu"]
    )
```

1. First, you create an artifact object (`wandb.Artifact()`).
2. Next, you add two dataset PyTorch tensors to the artifact object with `wandb.Artifact.add_file()`. 
3. Lastly, you link the artifact version to the `Model` collection in the `Zoo_Classifier_Models` registry with `link_artifact()`. You also add two custom aliases to the artifact version by passing  `production-us` and `production-eu` as arguments to the `aliases` parameter.