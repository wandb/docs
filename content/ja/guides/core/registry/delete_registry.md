---
menu:
  default:
    identifier: ja-guides-core-registry-delete_registry
    parent: registry
title: Delete registry
weight: 8
---

This page shows how a Team admin or Registry admin can delete a custom registry.  A [core registry]({{< relref path="/guides/core/registry/registry_types#core-registry" lang="ja" >}}) cannot be deleted.

- A Team admin can delete any custom registry in the organization.
- A Registry admin can delete a custom registry that they created.

Deleting a registry also deletes collections that belong to that registry, but does not delete artifacts linked to the registry.  Such an artifact remains in the original project that the artifact was logged to.


{{< tabpane text=true >}}
{{% tab header="Python SDK" value="python" %}}

Use the `wandb` API's `delete()` method to delete a registry programmatically.  The following example illustrates how to:

1. Fetch the registry you want to delete with `api.registry()`.
1. Call the `delete()` method on the returned registry object to delete the registry.

```python
import wandb

# Initialize the W&B API
api = wandb.Api()

# Fetch the registry you want to delete
fetched_registry = api.registry("<registry_name>")

# Deleting a registry
fetched_registry.delete()
```

{{% /tab %}}

{{% tab header="W&B App" value="app" %}}

1. Navigate to the **Registry** App at https://wandb.ai/registry/.
2. Select the custom registry you want to delete.
3. Click the gear icon in the upper right corner to view the registry's settings.
4. To delete the registry, click the trash can icon in the upper right corner of the settings page.
5. Confirm the registry to delete by entering its name in the modal that appears, then click **Delete**.

{{% /tab %}}
{{< /tabpane >}}