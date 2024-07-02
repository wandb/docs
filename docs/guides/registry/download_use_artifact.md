---
displayed_sidebar: default
---

# Download and use an artifact from a registry

Use the W&B Python SDK to download an artifact that you linked to the W&B Registry. Specify the artifact you want to download with the either of the following formats:


Replace values within `<>` with your own:

```python
import wandb

ENTITY = "<team-entity-name>"
PROJECT_NAME = "<project-name>"

ORG_ENTITY = '<organization-entity>'
REGISTRY_NAME = '<registry-name>'
COLLECTION_NAME = '<collection-name>'
ALIAS = '<artifact-alias>'
# INDEX = '<artifact-index>'

run = wandb.init(entity=ENTITY, project=PROJECT_NAME)

artifact_name = f"{ORG_ENTITY}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:{ALIAS}"
downloaded_path = run.use_artifact(artifact_or_name=name)
```

Reference an artifact with one of following formats listed:

```python
# Artifact name with index specified
f"{ORG_ENTITY}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:v{INDEX}"

# Artifact name with alias specified
f"{ORG_ENTITY}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:{ALIAS}"
```
Where:
* `latest` - Use `latest` alias to specify the version that is most recently linked.
* `v#` - Use `v0`, `v1`, `v2`, and so on to fetch a specific version in the collection.
* `alias` - Specify the custom alias attached to the artifact version

See [`use_artifact`](../../ref/python/run.md#use_artifact) in the API Reference guide for more information on possible parameters and return type.

<details>
<summary>Example: Download and use a linked model artifact</summary>

For example, in the proceeding code snippet a user called the `use_artifact` API. They specified the name of the model artifact they want to fetch and they also provided a version/alias. They then stored the path that returned from the API to the `downloaded_path` variable.

```python
import wandb


ENTITY_NAME = "smle-reg-team-2"
PROJECT_NAME = "registry_demo"

ORG_ENTITY_NAME = "smle-registries-bug-bash"
REGISTRY_NAME = "Fine-tuned models"
COLLECTION_NAME = "MNIST"
ALIAS = 'production'

# Initialize a run
run = wandb.init(entity=ENTITY_NAME, propject=PROJECT_NAME)

# Access and download artifact. Returns path to downloaded artifact
downloaded_path = run.use_artifact(artifact_or_name=f"{ORG_ENTITY_NAME}/wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}:{alias}")
```
</details>