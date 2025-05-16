---
menu:
  default:
    identifier: delete_registry
    parent: registry
title: Delete registry
weight: 8
---

Team or Registry admins can delete a custom registry. A team admin can delete any custom registry in the organization, while a registry admin can only delete the custom registry they created.

Deleting a registry also deletes collections that belong to that registry. You cannot delete a core registry.


This action is permanent and cannot be undone.

```python
import wandb

# Initialize the W&B API
api = wandb.Api(overrides={"organization": "registries-bug-bash"})

# Fetch the registry you want to delete
fetched_registry = api.registry("my-new-registry")

# Deleting a registry
fetched_registry.delete()
```