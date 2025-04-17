---
title: Use aliases to point to a specific artifact version
weight: 5
---


Use aliases to reference a specific artifact version. [W&B automatically assigns aliases]({{< relref "aliases#default-aliases" >}}) to each artifact you link with the same name. You can also [create one or more custom aliases]({{< relref "aliases#custom-aliases" >}}) to reference a specific artifact version.

{{% alert title="When to use a tag versus using an alias" %}}
Use aliases to reference a specific artifact version. Each alias within a collection is unique. Only one artifact version can have a specific alias at a time.

Use tags to organize and group artifact versions or collections based on a common theme. Multiple artifact versions and collections can share the same tag.
{{% /alert %}}

## Default aliases

W&B automatically assigns the following aliases to each artifact you link with the same name:

* `latest` to the most recent artifact version you link to a collection.
* Unique version number. W&B counts each artifact version you link and uses that count number to assign a unique version number to that artifact. Zero indexing is used.

For example, if you link an artifact named `zoo_model` three times, W&B creates three aliases `v0`, `v1`', and `v2` respectively.

## Custom aliases

Create custom aliases to reference a specific artifact versions based on your unique use case. As an example, you might use aliases `dataset_version_v0`, `dataset_version_v1`, `dataset_version_v2`, and so forth to identify which dataset a model was trained on. As an another example, you might use a `best_model` alias to keep track of the best performing artifact model version.

Any user with a [Member or Admin registry role]({{< relref "guides/core/registry/configure_registry/#registry-roles" >}}) can add or delete an alias from a linked artifact.

{{% alert%}}
<!-- Consider using a [protected alias]({{< relref "aliases/#protected-aliases" >}}) to prevent future modification or deletion of an artifact version. Only Registry Admins can add, modify, or remove protected aliases from a registry. -->
Consider using a [protected alias]({{< relref "aliases/#protected-aliases" >}}) to label and identify artifact versions that should not be modified or deleted.
{{% /alert %}}

You can create a custom alias with the W&B Registry or the Python SDK. Based on your use case, click on the tab below that best fits your needs.

{{< tabpane text=true >}}
{{% tab header="W&B Registry" value="app" %}}

1. Navigate to the W&B Registry.
2. Click the **View details** button in a collection.
3. Within the **Versions** section, select the **View** button for a specific artifact version.
4. Select the plus icon (**+**) to add one or more aliases next to the **Aliases** field.

{{% /tab %}}

{{% tab header="Python SDK" value="python" %}}
When you link an artifact version to a collection with the Python SDK you can optionally provide a list of one or more aliases as an argument to the `alias` parameter in [`link_artifact()`]({{< relref "/ref/python/run/#link_artifact" >}}).

The following code snippet demonstrates how to link an artifact version to a collection with the Python SDK. Replace values within `<>` with your own:

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

Only [registry admins]({{< relref "/guides/core/registry/configure_registry/#registry-roles" >}}) can add, modify, or remove protected aliases from a collection. See [Configure registry access]({{< relref "/guides/core/registry/configure_registry.md" >}}) for information on how to manage users and assign roles in a registry.

<!-- Protected aliases are not shared across registries.  -->

Common protected aliases include:

- **Production**: The artifact version is ready for production use.
- **Staging**: The artifact version is ready for testing.

#### Create protected aliases in a registry

1. Navigate to the Registry App.
2. Select a registry.
3. Click the gear button on the top right of the page to view the registry's settings.
4. Within the **Protected Aliases** section, click on the plus icon (**+**) to add one or more protected aliases.


<!-- ## Check for existing aliases

MongoDB query + UI -->