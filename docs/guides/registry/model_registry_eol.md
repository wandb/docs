---
displayed_sidebar: default
---

# Migrating from legacy Model Registry

W&B will migrate assets from the legacy [W&B Model Registry](../model_registry/intro.md) to [W&B Registry](./intro.md) in mid-October 2024.By the end of 2024 you will no longer be able to access W&B Model Registry. An exact date will be shared at least 3 weeks in advance to all W&B Model Registry users. 

This guide will cover what changes will occur, how W&B will migrate assets, along with information on how to prepare your current assets for the migration process.


Contact support@wandb.com for more information about the migration process.


## What is changing?

The following tables highlights the major changes that will happen from the legacy Model Registry to the W&B Registry:

|               | legacy W&B Model Registry | W&B Registry |
| -----         | ----- | ----- |
| Artifact visibility| Team level | Organization level |
| Name change | A set of pointers (links) to model versions are called *registered models*. | A set of pointers (links) to artifact versions are called *collections*. | 
| `wandb.init.link_model` | Model Registry specific API | Currently only compatible with legacy model registry |



<!-- Registered models in the legacy W&B Model Registry are renamed to Collections in W&B Registry. -->

<!-- Artifacts in the new W&B Registry have organization level scope. Artifacts in the legacy W&B Model Registry have team level scope.  -->


## Preparing for the migration

### What is migrating?

W&B will migrate [registered models](../model_registry/create-registered-model.md)(now called collections) and the [artifact versions](../model_registry/link-model-version.md) that are in your current W&B Model Registry, to W&B Registry as collections. 

![](/images/registry/eol_migration.png)

More specifically, your registered models(collections) will be stored within within a registry called **Model**.

![](/images/registry/mode_reg_eol.png)

:::info Team visibility to organization visibility
All registries in the legacy Model Registry, which have team-level visibility, will have organizational-level visibility after the migration.
:::



### Artifact path changes

After the migration the paths to artifacts linked within collections will change from:

```python
team-name/model-registry/collection-name:v0
```
to:

```python
org-name/wandb-registry-model/team-name_collection-name:v0
```

:::info Action required
Update any programmatic references to legacy Model Registry paths within your code to avoid disruptions. 
:::

For example, suppose that your current code looks like:

```python
import wandb

team_entity = "team_awesome"
project = "cool_project"
registered_model_name = "zoo-dataset-tensors"
version = "latest"

run = wandb.init(entity = team_entity, project = project)

# Create artifact object and add dataset to artifact
artifact = wandb.Artifact(name = "zoo_data", type = "dataset")
artifact.add_file(local_path = "zoo_data.pt", name = "dataset")

# The path where you want to link the artifact version to
target_path = f"{team_entity}/model-registry/{registered_model_name}:{version}"

run.link_artifact(artifact = artifact, target_path = target_path)
```

In preparation of the migration, you will need to ensure that you specify the entity of the organization and provide a name for the collection (formally known as registered model in legacy W&B Model Registry):

```python
import wandb

org_entity = "reviewco" # Use organization entity
project = "cool_project"
collection = "zoo-dataset-tensors" # Registered models now called collections
version = "latest"

run = wandb.init(entity = org_entity, project = project)

# Create artifact object and add dataset to artifact
artifact = wandb.Artifact(name = "zoo_data", type = "dataset")
artifact.add_file(local_path = "zoo_data.pt", name = "dataset")

target_path = f"{org_entity}/wandb-registry-model/{collection}:{version}"

# Link an artifact to a collection called zoo-dataset-tensors within the new model registry
run.link_artifact(artifact = artifact, target_path = target_path)
```

The proceeding table lists the parameters that you need to update the path name for:


| Method | Parameter | 
| -- | -- | 
| [`wandb.run.link_artifact()`](../../ref/python/run.md#link_artifact) | `target_path` | 
| [`wandb.Artifact.link()`](../../ref/python/artifact.md#link) |`target_path` | 
| [`wandb.Api().run.use_artifact()`](../../ref/python/public-api/run.md#use_artifact)| `artifact`| 
| [`wandb.Api().artifact()`](../../ref/python/public-api/api.md#artifact) | `name` | 


### Collection attributes

W&B will migrate the following attributes in your current registered models to collections:

- Descriptions
- Tags
- Aliases

### Automations

<!-- Noah to update automation links -->

Any automations that your team created will be migrated to the new registry. The automations will use the same registered model/collection that you created the automation for. 

### Deprecated features 

* [Protected aliases](../model_registry/access_controls.md#add-protected-aliases): Aliases will no longer have a protected status available. Instead, you can go to a registry’s settings to configure which users have viewer, member, or admin status. All aliases that were previously protected will become standard aliases. 
*  Service accounts:  Existing team service accounts cannot interact programmatically with Registry as the Registry is at the org level. As a result, Organization Service accounts will be made available (this feature is currently in progress) prior to the migration taking place. Please reach out to support@wandb.com to discuss temporary solutions that we recommend. 


## During the migration

Once the migration begins, there will be a write-block on the legacy Model Registry. This write-block means that users will not be able to do the following in the legacy Model Registry:

- Create  and delete collections (formerly known as registered models)
- Link new versions and unlink existing versions
- Add and delete aliases to artifacts in the registry
- Add and delete tags to collections in the registry
- Add and delete automations

The migration will last for approximately 24-48 hours and will be conducted during non-business hours to minimize disruption. The legacy Model Registry will remain in a read-only state after the migration.

## After the migration

After the migration all artifact versions, collections, descriptions, automations, and tags will be migrated to W&B Registry. Action history will be preserved. You will still be able to navigate to the action history prior to the migration. In addition, W&B will add a new action to represent which linked version are migrated.

### Using the new Registry
All organizations should plan to use the new model registry inside the new W&B Registry. The same capabilities will be supported, including automations, lineage, and aliasing, with new features introduced. Review the list in the [Deprecated features](#deprecated-features) section for more information about features that will no longer be supported.

By default, organization administrators will be admins of the new model registry and all other orgs users will be viewers to the model registry. To modify any user’s access to the registry - learn more [here](https://docs.wandb.ai/guides/registry/configure_registry#configure-user-roles-in-a-registry).


### Legacy Model Registry
After the migration, the legacy Model Registry UI will be read-only. It will eventually be fully deprecated from the W&B App UI. You will still have view-only access to all linked models in your legacy registry using the following paths:

1. See linked models in this view-only W&B Project:
```python
[WANDB_BASE_URL]/team_entity_name/model-registry/artifacts/
```
2. Programmatically reference a linked model with the following path:
```python
team_entity_name/model-registry/collection:alias
```


## FAQs

### Why is W&B migrating assets from Model Registry to Registry?
W&B is migrating assets from Model Registry to Registry to ensure that users can start using the new capabilities of the new W&B Registry and transition their existing contents before the legacy registry is deprecated. 

### What do I need to do before the migration?
Ensure that any programmatic references to model paths are updated to reflect the new path format post-migration. This [section contains](#artifact-path-changes) instructions on paths that require updates.

### Will I lose access to my model artifacts?
No, you will not lose access to the artifacts in the legacy Model Registry. You will continue to retain read access to the legacy paths (and have the ability to perform read actions with the W&B Python SDK). Writes and modifications to the legacy Model Registry will be restricted as described here.

### How can I get help if I encounter issues?
Reach out to support@wandb.com with any questions, concerns, or if you need assistance with updating your programmatic paths.