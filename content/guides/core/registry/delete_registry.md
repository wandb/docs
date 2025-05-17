---
menu:
  default:
    identifier: delete_registry
    parent: registry
title: Delete registry
weight: 8
---

Team or Registry admins can delete a custom registry. A team admin can delete any custom registry in the organization, while a registry admin can only delete the custom registry they created.

You cannot delete a core registry. Deleting a registry also deletes collections that belong to that registry. The registry's artifacts are not deleted, but the artifacts are no longer associated with the registry.


{{< tabpane text=true >}}
{{% tab header="Python SDK" value="python" %}}

Use the `wandb.Api` to delete a registry programmatically. First, fetch the registry you want to delete with `api.registry()`, then call the returned registry object's `delete` method to delete the registry.

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
3. Click on the gear icon in the upper right corner to view the registry's settings.
4. Click on the trash can icon in the upper right corner of the settings page.
5. Specify the name of the registry in the pop-up window to confirm that you want to delete the registry. Click **Delete**.

{{% /tab %}}
{{< /tabpane >}}