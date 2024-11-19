---
slug: /guides/registry
displayed_sidebar: default
title: Registry
---

:::info
W&B Registry is in public preview. See [this](#enable-wb-registry) section to learn how to enable it for your deployment type.


The legacy Model Registry is scheduled for deprecation with the exact date not yet decided. See [Migrating from legacy Model Registry](./model_registry_eol.md) for more information about the migration process.
:::

<!-- W&B Registry is a curated central repository of artifact versions within your organization. Teams across your organization can share and collaboratively manage the lifecycle of all artifacts such as models and datasets. 

Registry provides the foundation for an effective CI/CD pipeline by identifying the right models to reproduce, retrain, evaluate, and deploy

You can use the Registry to version artifacts, audit the history of an artifact's usage and changes, ensure governance and compliance of your artifacts such as training models and datasets. -->


W&B Registry is a curated central repository of [artifact](../artifacts/intro.md) versions within your organization. Users who [have permission](./configure_registry.md) within your organization can [download](./download_use_artifact.md), share, and collaboratively manage the lifecycle of all artifacts, regardless of the team that user belongs to.

You can use the Registry to [track artifact versions](./link_version.md), audit the history of an artifact's usage and changes, ensure governance and compliance of your artifacts, and [automate downstream processes such as model CI/CD](../automations/intro.md).

In summary, use W&B Registry to:

- [Promote](./link_version.md) artifact versions that satisfy a machine learning task to other users in your organization.
- Organize [artifacts with tags](./organize-with-tags.md) so that you can find or reference specific artifacts.
- Track an [artifact’s lineage](../model_registry/model-lineage.md) and audit the history of changes.
- [Automate](../model_registry/model-registry-automations.md) downstream processes such as model CI/CD.
- [Limit who in your organization](./configure_registry.md) can access artifacts in each registry.

<!-- - Quickly find or reference important artifacts with a unique identifier known as aliases.-->

![](/images/registry/registry_landing_page.png)

The preceding image shows the Registry App with "Model" and "Dataset" core registries along with custom registries.


## How it works
Each organization initially contains two registries that you can use to organize your model and dataset artifacts called "Models" and "Datasets", respectively. You can create [additional registries to organize artifacts based on your organization's needs](./registry_types.md). 

Each [registry](./configure_registry.md) consists of one or more [collections](./create_collection.md). Each collection represents a distinct tasks or use cases.

To add an artifact to a registry, you first log a specific artifact version to W&B. Once you log the artifact version, you can link that artifact version to a collection in the registry.

As an example, the proceeding code snippet demonstrates how to log and link a fake model artifact called "my_model.txt" to a collection named "new-collection" in the [default "Models" registry](./registry_types.md). To do this, you need to:

1. Log an artifact version to W&B with `wandb.run.init()`.
2. Specify the name of the collection you want to link your artifact version to.
2. Link the artifact version to the registry.

Copy and paste the proceeding code snippet into a Python script and run it to log and link an artifact version to the **Models** registry:

```python title="simply_registry_example.py"
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
```


Once you have an artifact version in the registry, you can use the Registry App to view, download, and manage the artifact version, create downstream automations such as model CI/CD workflows, and more.


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



