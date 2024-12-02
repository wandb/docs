---
menu:
  default:
    identifier: download_use_artifact
    parent: registry
title: Download and use an artifact from a registry
---

Use the W&B Python SDK to use and download an artifact that you linked to the W&B Registry. 

:::note
To find the usage code snippets for a specific artifact pre-populated with the path information, please refer to the section [Copy and paste the usage path from the Registry UI](#copy-and-paste-the-usage-path-from-the-registry-ui).
:::

Replace values within `<>` with your own:

```python
import wandb

ORG_ENTITY_NAME = '<org-entity-name>'
REGISTRY_NAME = '<registry-name>'
COLLECTION_NAME = '<collection-name>'
ALIAS = '<artifact-alias>'
INDEX = '<artifact-index>'

run = wandb.init()  # Optionally use the entity, project arguments to specify where the run should be created

registered_artifact_name = f"{ORG_ENTITY_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:{ALIAS}"
registered_artifact = run.use_artifact(artifact_or_name=name)  # marks this artifact as an input to your run
artifact_dir = registered_artifact.download()  
```

Reference an artifact version with one of following formats listed:

```python
# Artifact name with version index specified
f"{ORG_ENTITY}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{INDEX}"

# Artifact name with alias specified
f"{ORG_ENTITY}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:{ALIAS}"
```
Where:
* `latest` - Use `latest` alias to specify the version that is most recently linked.
* `v#` - Use `v0`, `v1`, `v2`, and so on to fetch a specific version in the collection.
* `alias` - Specify the custom alias attached to the artifact version

See [`use_artifact`](../../ref/python/run.md#use_artifact) and [`download`](/ref/python/artifact#download) in the API Reference guide for more information on possible parameters and return type.

<details>
<summary>Example: Use and download an artifact linked to the W&B Registry</summary>

For example, in the proceeding code snippet a user called the `use_artifact` API. They specified the name of the model artifact they want to fetch and they also provided a version/alias. They then stored the path that returned from the API to the `downloaded_path` variable.

```python
import wandb
TEAM_NAME = "product-team-applications"
PROJECT_NAME = "user-stories"

ORG_ENTITY_NAME = "wandb"
REGISTRY_NAME = "Fine-tuned Models"
COLLECTION_NAME = "phi3-finetuned"
ALIAS = 'production'

# Initialize a run inside the specified team and project
run = wandb.init(entity=TEAM_NAME, propject=PROJECT_NAME)

registered_artifact_name = f"{ORG_ENTITY_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:{ALIAS}"

# Access an artifact and mark it as input to your run for lineage tracking
registered_artifact = run.use_artifact(artifact_or_name=name)  # 
# Download artifact. Returns path to downloaded contents
downloaded_path = registered_artifact.download()  
```
</details>

## Copy and paste the usage path from the Registry UI

You can also find the exact code snippet to use and download a specific artifact version inside the Usage tab in the Registry UI to avoid constructing the path yourself. The required fields will be populated based on the which artifact version's details you are viewing:

1. **Navigate to the W&B Registry:**
   - Go to the "Registry" tab in the sidebar to access the list of registries.

2. **Select the Desired Registry:**
   - From the list of registries, click on the registry that contains the artifact you want to use. 

3. **Find the Artifact Collection:**
   - In the registry details page, locate the collection that includes your desired artifact. Click on the collection name to view its versions.

4. **Access the Usage Tab:**
   - Click on the version of the artifact you need. This will open the artifact version details page.
   - On the artifact version details page, switch to the "Usage" tab.

5. **Copy the Code Snippet:**
   - In the "Usage" tab, you will see code snippets for using and downloading the artifact. These snippets are pre-filled with the correct path to the artifact.
   - Copy the relevant code snippet for your use case. The code snippet will look something like this:

   ```python
   import wandb

   run = wandb.init()

   artifact = run.use_artifact('registries-bug-bash/wandb-registry-model/registry-quickstart-collection:v3', type='model')
   artifact_dir = artifact.download()
   ```
   ![](/images/registry/find_usage_in_registry_ui.gif)