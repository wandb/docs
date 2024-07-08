---
description: Update an existing Artifact inside and outside of a W&B Run.
displayed_sidebar: default
---

# Update artifacts

<head>
  <title>Update artifacts</title>
</head>

Pass desired values to update the `description`, `metadata`, and `alias` of an artifact. Call the `save()` method to update the artifact on the W&B servers. You can update an artifact during a W&B Run or outside of a Run.

Use the W&B Public API ([`wandb.Api`](../../ref/python/public-api/api.md)) to update an artifact outside of a run. Use the Artifact API ([`wandb.Artifact`](../../ref/python/artifact.md)) to update an artifact during a run.

:::caution
You can not update the alias of artifact that is linked to a model in Model Registry.
:::


import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs
  defaultValue="duringrun"
  values={[
    {label: 'During a Run', value: 'duringrun'},
    {label: 'Outside of a Run', value: 'outsiderun'},
  ]}>
  <TabItem value="duringrun">

The proceeding code example demonstrates how to update the description of an artifact using the [`wandb.Artifact`](../../ref/python/artifact.md) API:

```python
import wandb

run = wandb.init(project="<example>", job_type="<job-type>")
artifact = run.use_artifact("<artifact-name>:<alias>")

run.use_artifact(artifact)
artifact.description = "<description>"
artifact.save()
```
  </TabItem>
  <TabItem value="outsiderun">

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
  </TabItem>
</Tabs>
