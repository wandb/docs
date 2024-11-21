---
displayed_sidebar: default
title: Download and use an artifact from a registry
---

Use the W&B Python SDK to download an artifact linked to a registry. In order to download and use an artifact, you need to know the registry name, collection name, and the alias or index of the artifact version you want to access.




<!-- Find the path of the artifact prgrammatically or with the SDK -->
<!-- smle-registries-bug-bash/wandb-registry-model/Zoo_Classifier_Models:v4 -->


## Download an artifact linked to the W&B Registry programmatically

To download an artifact linked to a registry, you must know the registry name, collection name, and the alias or index of the artifact version you want to access. 

You use these properties to construct the path to the linked artifact:

```python
# Artifact name with version index specified
f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{INDEX}"

# Artifact name with alias specified
f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:{ALIAS}"
```

You can find the full path of the artifact version in the W&B App UI:

1. Navigate to the Registry App.
2. Select the name of the registry that contains your artifact.
3. Select the name of the collection.
4. From the list of artifact versions, select the version you want to access.
5. Select the **Version** tab.
6. Copy and paste the name of the artifact shown in the **Full Name** field.

<!-- smle-registries-bug-bash/wandb-registry-model/Zoo_Classifier_Models:v4 -->


The proceeding code snippet shows how to use and download an artifact linked to the W&B Registry. Ensure to replace values within `<>` with your own:

```python
import wandb

REGISTRY_NAME = '<registry-name>'
COLLECTION_NAME = '<collection-name>'
ALIAS = '<artifact-alias>'

run = wandb.init(
   entity = '<team-name>',
   project = '<project-name>'
   )  

artifact_name = f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:{ALIAS}"
# artifact_name = '<artifact_name>' # Copy and paste from the W&B App UI
registered_artifact = run.use_artifact(artifact_or_name = name)  
download_path = registered_artifact.download()  
```

:::tip
Use the `latest` alias to specify the version that is most recently linked.
:::


<details>
<summary>Example: Use and download an artifact linked to the W&B Registry</summary>

The proceeding code example shows a user downloading an artifact linked to a collection called "phi3-finetuned" in the "Fine-tuned Models" registry. The alias of the artifact version is set to "production".

```python
import wandb

TEAM_ENTITY = "product-team-applications"
PROJECT_NAME = "user-stories"

REGISTRY_NAME = "Fine-tuned Models"
COLLECTION_NAME = "phi3-finetuned"
ALIAS = 'production'

# Initialize a run inside the specified team and project
run = wandb.init(entity=TEAM_ENTITY, project = PROJECT_NAME)

artifact_name = f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:{ALIAS}"

# Access an artifact and mark it as input to your run for lineage tracking
registered_artifact = run.use_artifact(artifact_or_name = name)  

# Download artifact. Returns path to downloaded contents
downloaded_path = registered_artifact.download()  
```
</details>

See [`use_artifact`](../../ref/python/run.md#use_artifact) and [`Artifact.download()`](/ref/python/artifact#download) in the API Reference guide for more information on possible parameters and return type.

## Download an artifact linked to the W&B Registry interactively

Copy and paste the usage path from the Registry UI

You can also find the exact code snippet to use and download a specific artifact version inside the **Usage** tab in the Registry UI to avoid constructing the path yourself. The required fields are populated based on the artifact version.

1. Navigate to the Registry App.
2. Select the name of the registry that contains your artifact.
3. Select the name of the collection.
4. From the list of artifact versions, select the version you want to access.
5. Select the **Usage** tab.
6. Copy the code snippet shown in the **Usage API** section.
7. Paste the code snippet into your Python script, notebook, or terminal.

![](/images/registry/find_usage_in_registry_ui.gif)
