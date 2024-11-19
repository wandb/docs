---
slug: /guides/registry
displayed_sidebar: default
title: Registry
---

:::info
W&B Registry is in public preview. Visit [this](#enable-wb-registry) section to learn how to enable it for your deployment type.
:::

W&B Registry is a curated central repository of artifact versions within your organization. Teams across your organization can share and collaboratively manage the lifecycle of all artifacts such as models and datasets.


![](/images/registry/registry_landing_page.png)

Use W&B Registry to:

- [Bookmark](./link_version.md) your best artifacts for each machine learning task.
- [Automate](../model_registry/model-registry-automations.md) downstream processes and model CI/CD.
- Track an [artifact’s lineage](../model_registry/model-lineage.md) and audit the history of changes to production artifacts.
- [Configure](./configure_registry.md) viewer, member, or administrator access to a registry for all organization users.
- Quickly find or reference important artifacts with a unique identifier known as aliases.
- Use [tags](./organize-with-tags.md) to label, group, and discover assets in your Registry. 

## How it works
Each organization initially contains two registries that you can use to organize your model artifacts and dataset artifacts called "Models" and "Datasets", respectively. You can create [additional registries to organize artifacts based on your organization's needs](./registry_types.md). 


Each [registry](./configure_registry.md) consists of one or more [collections](./create_collection.md). Each collection represents a distinct tasks or use cases.


As an example, the proceeding code snippet demonstrates how to log and link a fake artifact model called "my_model.txt" to a collection named "new-collection" in the [default (or core) "Models" registry](./registry_types.md). To do this, you need to:

1. Log an artifact version to W&B with `wandb.run.init()`.
2. Specify the name of the collection you want to link your artifact version to.
2. Link the artifact version to the registry.

Copy and paste the proceeding code snippet into a Python script and run it to log and link an artifact version to the **Models** registry:

```python
import wandb
import random

# Start a new W&B run to track your experiment
run = wandb.init(project="registry_quickstart") 

# Create a simulated model file
with open("my_model.txt", "w") as f:
   f.write("Model: " + str(random.random()))

# Log the model to W&B
logged_artifact = run.log_artifact(
    artifact_or_path="./my_model.txt", 
    name="gemma-finetuned", 
    type="model" # Specifies artifact type
    )

# Specify the name of the collection and registry
# you want to publish the artifact to
COLLECTION_NAME = "new-collection"
REGISTRY_NAME = "model"

# Link the artifact to the registry
run.link_artifact(
    artifact=logged_artifact, 
    target_path=f"{REGISTRY_NAME}/{COLLECTION_NAME}"
    )

run.finish()
```


## Enable W&B Registry

Based on your deployment type, satisfy the following conditions to enable W&B Registry:

| Deployment type | How to enable |
| ----- | ----- |
| Multi-tenant Cloud | No action required. W&B Registry is available on the W&B App. |
| Dedicated Cloud | Contact your account team. The Solutions Architect (SA) Team enables W&B Registry within your instance's operator console. Ensure your instance is on server release version 0.59.2 or newer.|
| Self-Managed   | Enable the environment variable called `ENABLE_REGISTRY_UI`. To learn more about enabling environment variables in server, visit [these docs](/guides/hosting/env-vars). In self-managed instances, your infrastructure administrator should enable this environment variable and set it to `true`. Ensure your instance is on server release version 0.59.2 or newer.|


## Resources to get started

Depending on your use case, explore the following resources to get started with the W&B Registry:

* Check out the tutorial video:
    * [Getting started with Registry from Weights & Biases](https://www.youtube.com/watch?v=p4XkVOsjIeM)
* Take the W&B [Model CI/CD](https://www.wandb.courses/courses/enterprise-model-management) course and learn how to:
    * Use W&B Registry to manage and version your artifacts, track lineage, and promote models through different lifecycle stages.
    * Automate your model management workflows using webhooks.
    * See how Registry integrates with external ML systems and tools in your model development lifecycle for model evaluation, monitoring, and deployment.

## Migrating from the legacy Model Registry to W&B Registry

The legacy Model Registry is scheduled for deprecation with the exact date not yet decided. Before deprecating the legacy Model Registry, W&B will migrate the contents of the legacy Model Registry to the W&B Registry. 


See [Migrating from legacy Model Registry](./model_registry_eol.md) for more information about the migration process from the legacy Model Registry to W&B Registry.

Until the migration occurs, W&B supports both the legacy Model Registry and the new Registry. 

:::info
To view the legacy Model Registry, navigate to the Model Registry in the W&B App. A banner appears at the top of the page that enables you to use the legacy Model Registry App UI.

![](/images/registry/nav_to_old_model_reg.gif)
:::


Reach out to support@wandb.com with any questions or to speak to the W&B Product Team about any concerns about the migration.


