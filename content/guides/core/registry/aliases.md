---
title: Use aliases to point to a specific artifact version
weight: 5
---

Reference a specific artifact version with an alias. An artifact version can have one or more aliases. [W&B automatically assigns aliases]({{< relref "aliases#default-aliases" >}}) to each artifact you link with the same name. You can also [create one or more custom aliases]({{< relref "aliases#custom-aliases" >}}) to reference a specific artifact version.

Aliases appear as rectangles with the name of that alias in the rectangle in the Registry UI. If an [alias is protected]({{< relref "aliases#protected-aliases" >}}), it appears as a gray rectangle with a lock icon. Otherwise, the alias appears as an orange rectangle. Aliases are not shared across registries.

{{% alert title="When to use a tag versus using an alias" %}}
Use aliases to reference a specific artifact version. Each alias within a collection is unique. Only one artifact version can have a specific alias at a time.

Use tags to organize and group artifact versions or collections based on a common theme. Multiple artifact versions and collections can share the same tag.
{{% /alert %}}

## Default aliases

W&B automatically assigns the following aliases to each artifact version you link with the same name:

* The `latest` alias to the most recent artifact version you link to a collection.
* A unique version number. W&B counts each artifact version (zero indexing) you link. W&B uses the count number to assign a unique version number to that artifact.

For example, if you link an artifact named `zoo_model` three times, W&B creates three aliases `v0`, `v1`', and `v2` respectively.

## Custom aliases

Create custom aliases to reference a specific artifact versions based on your unique use case. As an example, you might use aliases `dataset_version_v0`, `dataset_version_v1`, `dataset_version_v2`, and so forth to identify which dataset a model was trained on. As an another example, you might use a `best_model` alias to keep track of the best performing artifact model version.

Any user with a [Member or Admin registry role]({{< relref "guides/core/registry/configure_registry/#registry-roles" >}}) can add or remove a custom alias from a linked artifact. Consider using a [protected alias]({{< relref "aliases/#protected-aliases" >}}) to label and identify artifact versions that should not be modified or deleted.

You can create a custom alias with the W&B Registry or the Python SDK. Based on your use case, click on a tab below that best fits your needs.

{{< tabpane text=true >}}
{{% tab header="W&B Registry" value="app" %}}

1. Navigate to the W&B Registry.
2. Click the **View details** button in a collection.
3. Within the **Versions** section, select the **View** button for a specific artifact version.
4. Select the plus icon (**+**) to add one or more aliases next to the **Aliases** field.

{{% /tab %}}

{{% tab header="Python SDK" value="python" %}}
When you link an artifact version to a collection with the Python SDK you can optionally provide a list of one or more aliases as an argument to the `alias` parameter in [`link_artifact()`]({{< relref "/ref/python/run/#link_artifact" >}}). W&B creates an alias for you if the alias you provide does not already exist.

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

Only [registry admins]({{< relref "/guides/core/registry/configure_registry/#registry-roles" >}}) can create protected aliases and add or remove protected aliases from an artifact version. See [Configure registry access]({{< relref "/guides/core/registry/configure_registry.md" >}}) for information on how to manage users and assign roles in a registry.

<!-- Admins can only create protected aliases in the W&B Registry UI. Once created, an admin can add a protected alias to an artifact version with the W&B Registry UI or the Python SDK. -->

<!-- Protected aliases are not shared across registries.  -->

Common protected aliases include:

- **Production**: The artifact version is ready for production use.
- **Staging**: The artifact version is ready for testing.

#### Create protected aliases

The following steps describe how to create a protected alias in the W&B Registry UI:

1. Navigate to the Registry App.
2. Select a registry.
3. Click the gear button on the top right of the page to view the registry's settings.
4. Within the **Protected Aliases** section, click on the plus icon (**+**) to add one or more protected aliases.

If successful, you will see one or more protected aliases appear as a gray rectangle with a lock icon in the **Protected Aliases** section. 

{{% alert %}}
Unlike custom non protected aliases, protected aliases can only be created in the W&B Registry UI and not programmatically with the Python SDK. Once created, admins can add a protected alias to an artifact version with the W&B Registry UI or the Python SDK.
{{% /alert %}}

The following steps describe how to add a protected alias to an artifact version with the W&B Registry UI:

1. Navigate to the W&B Registry.
2. Click the **View details** button in a collection.
3. Within the **Versions** section, select the **View** button for a specific artifact version.
4. Select the plus icon (**+**) to add one or more protected aliases next to the **Aliases** field.

Admins can add a protected alias to an artifact version programmatically with the Python SDK. Make sure that the protected alias already exists in the registry.

