---
description: >-
  Delete artifacts interactively with the App UI or programmatically with the
  W&B SDK/
displayed_sidebar: default
---

# Delete artifacts

<head>
  <title>Delete W&B Artifacts</title>
</head>

Delete artifacts interactively with the App UI or programmatically with the W&B SDK. When you delete an artifact, W&B marks that artifact as a *soft-delete*. In other words, the artifact is marked for deletion but files are not immediately deleted from storage. 

The contents of the artifact remain as a soft-delete, or pending deletion state, until a regularly run garbage collection job reviews all artifacts marked for deletion. The garbage collection algorithm deletes associated files from storage if the artifact and its associated files are not used by a previous or subsequent artifact versions. 


:::note
Garbage collection is not automatically enabled for W&B Server. Satisfy the following requirements to enable garbage collection in W&B Server:
* Set the `GORILLA_ARTIFACT_GC_ENABLED` environment variable to true: `GORILLA_ARTIFACT_GC_ENABLED=true`
* Enable bucket versioning if you use [AWS](https://docs.aws.amazon.com/AmazonS3/latest/userguide/manage-versioning-examples.html) or [GCP](https://cloud.google.com/storage/docs/object-versioning).
* [Enable soft delete for blobs if you use Azure](https://learn.microsoft.com/en-us/azure/storage/blobs/soft-delete-blob-overview).

W&B suggests that you set the number of bucket version copies to 1 because it helps reduce storage costs and can help recover accidentally deleted data.
:::

The sections in this page describe how to delete specific artifact versions, how to delete an artifact collection, how to delete artifacts with and without aliases, and more. You cam schedule when artifacts are deleted from W&B with TTL policies. For more information, see [Manage data retention with Artifact TTL policy](./ttl.md).

:::tip
It is recommended that you remove aliases associated to the artifact you want to delete before you delete that artifact. See the [Update an artifact](./update-an-artifact.md) documentation for information on how to programmatically or interactively update an alias with the W&B SDK or App UI, respectively.
:::

### Delete an artifact version

To delete an artifact version:

1. Select the name of the artifact. This will expand the artifact view and list all the artifact versions associated with that artifact.
2. From the list of artifacts, select the artifact version you want to delete.
3. On the right hand side of the workspace, select the kebab dropdown.
4. Choose Delete.

An artifact version can also be deleted programatically via the [delete()](https://docs.wandb.ai/ref/python/artifact#delete) method. See the examples below. 

### Delete multiple artifact versions with aliases

The following code example demonstrates how to delete artifacts that have aliases associated with them. Provide the entity, project name, and run ID that created the artifacts.

```python
import wandb

run = api.run("entity/project/run_id")

for artifact in run.logged_artifacts():
    artifact.delete()
```

Set the `delete_aliases` parameter to the boolean value, `True` to delete aliases if the artifact has one or more aliases.

```python
import wandb

run = api.run("entity/project/run_id")

for artifact in run.logged_artifacts():
    # Set delete_aliases=True in order to delete
    # artifacts with one more aliases
    artifact.delete(delete_aliases=True)
```

### Delete multiple artifact versions with a specific alias

The proceeding code demonstrates how to delete multiple artifact versions that have a specific alias. Provide the entity, project name, and run ID that created the artifacts. Replace the deletion logic with your own:

```python
import wandb

runs = api.run("entity/project_name/run_id")

# Delete artifact ith alias 'v3' and 'v4
for artifact_version in runs.logged_artifacts():
    # Replace with your own deletion logic.
    if artifact_version.name[-2:] == "v3" or artifact_version.name[-2:] == "v4":
        artifact.delete(delete_aliases=True)
```

### Delete all versions of an artifact that do not have an alias

The following code snippet demonstrates how to delete all versions of an artifact that do not have an alias. Provide the name of the project and entity for the `project` and `entity` keys in `wandb.Api`, respectively. Replace the `<>` with the name of your artifact:

```python
import wandb

# Provide your entity and a project name when you
# use wandb.Api methods.
api = wandb.Api(overrides={"project": "project", "entity": "entity"})

artifact_type, artifact_name = "<>"  # provide type and name
for v in api.artifact_versions(artifact_type, artifact_name):
    # Clean up versions that don't have an alias such as 'latest'.
    # NOTE: You can put whatever deletion logic you want here.
    if len(v.aliases) == 0:
        v.delete()
```

### Delete an artifact collection

To delete an artifact collection:

1. Navigate to the artifact collection you want to delete and hover over it.
3. Select the kebab dropdown next to the artifact collection name.
4. Choose Delete.

You can also delete artifact version programmatically with the [delete()](../../ref/python/artifact.md#delete) method. Provide the name of the project and entity for the `project` and `entity` keys in `wandb.Api`, respectively. Replace the `<>` with the name of your artifact:

```python
import wandb

# Provide your entity and a project name when you
# use wandb.Api methods.
api = wandb.Api(overrides={"project": "project", "entity": "entity"})

artifact_name = "<>"  # provide artifact name
artifact = api.artifact(artifact_name)
artifact.collection.delete()
```
