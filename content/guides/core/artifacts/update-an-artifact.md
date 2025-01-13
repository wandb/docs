---
description: Update an existing Artifact inside and outside of a W&B Run.
menu:
  default:
    identifier: update-an-artifact
    parent: artifacts
title: Update an artifact
weight: 4
---

Pass desired values to update the `description`, `metadata`, and `alias` of an artifact. Call the `save()` method to update the artifact on the W&B servers. You can update an artifact during a W&B Run or outside of a Run.

Use the W&B Public API ([`wandb.Api`](../../ref/python/public-api/api.md)) to update an artifact outside of a run. Use the Artifact API ([`wandb.Artifact`](../../ref/python/artifact.md)) to update an artifact during a run.

{{% alert color="secondary" %}}
You can not update the alias of artifact linked to a model in Model Registry.
{{% /alert %}}

{{< tabpane text=true >}}
  {{% tab header="During a run" %}}

The proceeding code example demonstrates how to update the description of an artifact using the [`wandb.Artifact`](../../ref/python/artifact.md) API:

```python
import wandb

run = wandb.init(project="<example>")
artifact = run.use_artifact("<artifact-name>:<alias>")
artifact.description = "<description>"
artifact.save()
```  
  {{% /tab %}}
  {{% tab header="Outside of a run" %}}
The proceeding code example demonstrates how to update the description of an artifact using the `wandb.Api` API:

```python
import wandb

api = wandb.Api()

artifact = api.artifact("entity/project/artifact:alias")

# Update the description
artifact.description = "My new description"

# Selectively update metadata keys
artifact.metadata["oldKey"] = "new value"

# Replace the metadata entirely
artifact.metadata = {"newKey": "new value"}

# Add an alias
artifact.aliases.append("best")

# Remove an alias
artifact.aliases.remove("latest")

# Completely replace the aliases
artifact.aliases = ["replaced"]

# Persist all artifact modifications
artifact.save()
```

For more information, see the Weights and Biases [Artifact API](../../ref/python/artifact.md).  
  {{% /tab %}}
  {{% tab header="With collections" %}}
You can also update an Artifact collection in the same way as a singular artifact:

```python
import wandb
run = wandb.init(project="<example>")
api = wandb.Api()
artifact = api.artifact_collection(type="<type-name>", collection="<collection-name>")
artifact.name = "<new-collection-name>"
artifact.description = "<This is where you'd describe the purpose of your collection.>"
artifact.save()
```
For more information, see the [Artifacts Collection](../../ref/python/public-api/api.md) reference.
  {{% /tab %}}
{{% /tabpane %}}

