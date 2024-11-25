---
displayed_sidebar: default
title: Download and use an artifact from a registry
---

Use the W&B Python SDK to download an artifact linked to a registry. In order to download and use an artifact, you need to know the registry name, collection name, and the alias or index of the artifact version you want to access.


## Download an artifact linked to the W&B Registry programmatically

To download an artifact linked to a registry, you must know the registry name, collection name, and the alias or index of the artifact version you want to access. 

Use the properties that describe your artifact  to construct the path to the linked artifact:

```python
# Artifact name with version index specified
f"wandb-registry-{REGISTRY}/{COLLECTION}:v{INDEX}"

# Artifact name with alias specified
f"wandb-registry-{REGISTRY}/{COLLECTION}:{ALIAS}"
```

For core registries such as the Model and Dataset registry, specify the registry name in the path as "model" or "dataset" respectively.


If you are not sure about the registry name, collection name, or alias of the artifact version you want to access, you can find this information in the W&B App UI.

1. Navigate to the Registry App.
2. Select the name of the registry that contains your artifact.
3. Select the name of the collection.
4. From the list of artifact versions, select the version you want to access.
5. Select the **Version** tab.
6. Copy and paste the name of the artifact shown in the **Full Name** field.



The proceeding code snippet shows how to use and download an artifact linked to the W&B Registry. Ensure to replace values within `<>` with your own:

```python
import wandb

REGISTRY = '<registry_name>'
COLLECTION = '<collection_name>'
ALIAS = '<artifact_alias>'

run = wandb.init(
   entity = '<team_name>',
   project = '<project_name>'
   )  

artifact_name = f"wandb-registry-{REGISTRY}/{COLLECTION}:{ALIAS}"
# artifact_name = '<artifact_name>' # Copy and paste Full name specified on the Registry App
fetched_artifact = run.use_artifact(artifact_or_name = artifact_name)  
download_path = fetched_artifact.download()  
```

Note that by using the `use_artifact` method, you are marking the artifact as an input to your run for lineage tracking. This allows you to track the lineage of the artifact in the W&B App UI. You can also download an artifact without creating a run by using the `wandb.Api()` object:

```python
import wandb

REGISTRY = "<registry_name>"
COLLECTION = "<collection_name>"
VERSION = "<version>"

api = wandb.Api()
artifact_name = f"wandb-registry-{REGISTRY}/{COLLECTION}:{VERSION}"
artifact = api.artifact(name = artifact_name)
```


<details>
<summary>Example: Use and download an artifact linked to the W&B Registry</summary>

The proceeding code example shows a user downloading an artifact linked to a collection called "phi3-finetuned" in the "Fine-tuned Models" registry. The alias of the artifact version is set to "production".

```python
import wandb

TEAM_ENTITY = "product-team-applications"
PROJECT_NAME = "user-stories"

REGISTRY = "Fine-tuned Models"
COLLECTION = "phi3-finetuned"
ALIAS = 'production'

# Initialize a run inside the specified team and project
run = wandb.init(entity=TEAM_ENTITY, project = PROJECT_NAME)

artifact_name = f"wandb-registry-{REGISTRY}/{COLLECTION}:{ALIAS}"

# Access an artifact and mark it as input to your run for lineage tracking
fetched_artifact = run.use_artifact(artifact_or_name = name)  

# Download artifact. Returns path to downloaded contents
downloaded_path = fetched_artifact.download()  
```
</details>



See [`use_artifact`](../../ref/python/run.md#use_artifact) and [`Artifact.download()`](/ref/python/artifact#download) in the API Reference guide for more information on possible parameters and return type.

:::note Users with a personal entity that belong to multiple organizations
Users with a personal entity that belong to multiple organizations must also specify either the name of their organization or use a team entity when accessing artifacts linked to a registry.

```python
import wandb

REGISTRY = "<registry_name>"
COLLECTION = "<collection_name>"
VERSION = "<version>"

# Ensure you are using your team entity to instantiate the API
api = wandb.Api(overrides={"entity": "<team-entity>"})
artifact_name = f"wandb-registry-{REGISTRY}/{COLLECTION}:{VERSION}"
artifact = api.artifact(name = artifact_name)

# Use org display name or org entity in the path
api = wandb.Api()
artifact_name = f"{ORG_ENTITY}/wandb-registry-{REGISTRY}/{COLLECTION}:{VERSION}"
artifact = api.artifact(name = artifact_name)
```
:::

## Download an artifact linked to the W&B Registry interactively

W&B creates a code snippet that you can copy and paste into your Python script, notebook, or terminal to download an artifact linked to a registry.

1. Navigate to the Registry App.
2. Select the name of the registry that contains your artifact.
3. Select the name of the collection.
4. From the list of artifact versions, select the version you want to access.
5. Select the **Usage** tab.
6. Copy the code snippet shown in the **Usage API** section.
7. Paste the code snippet into your Python script, notebook, or terminal.

![](/images/registry/find_usage_in_registry_ui.gif)
