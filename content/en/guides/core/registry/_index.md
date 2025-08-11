---
menu:
  default:
    identifier: registry
    parent: core
title: Registry
weight: 3
url: guides/core/registry
cascade:
- url: guides/core/registry/:filename
---
{{< cta-button colabLink="https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb_registry/zoo_wandb.ipynb" >}}

W&B Registry is a curated central repository of [W&B Artifact]({{< relref "/guides/core/artifacts/" >}}) versions within your organization. Users who [have permission]({{< relref "./configure_registry.md" >}}) within your organization can [download and use artifacts]({{< relref "./download_use_artifact.md" >}}), share, and collaboratively manage the lifecycle of all artifacts, regardless of the team that user belongs to.

You can use the Registry to [track artifact versions]({{< relref "./link_version.md" >}}), audit the history of an artifact's usage and changes, ensure governance and compliance of your artifacts, and [automate downstream processes such as model CI/CD]({{< relref "/guides/core/automations/" >}}).

In summary, use W&B Registry to:

- [Promote]({{< relref "./link_version.md" >}}) artifact versions that satisfy a machine learning task to other users in your organization.
- Organize [artifacts with tags]({{< relref "./organize-with-tags.md" >}}) so that you can find or reference specific artifacts.
- Track an [artifact’s lineage]({{< relref "/guides/core/registry/lineage.md" >}}) and audit the history of changes.
- [Automate]({{< relref "/guides/core/automations/" >}}) downstream processes such as model CI/CD.
- [Limit who in your organization]({{< relref "./configure_registry.md" >}}) can access artifacts in each registry.

<!-- - Quickly find or reference important artifacts with a unique identifier known as aliases.-->

{{< img src="/images/registry/registry_landing_page.png" alt="W&B Registry" >}}

The preceding image shows the Registry App with "Model" and "Dataset" core registries along with custom registries.


## Learn the basics
Each organization initially contains two registries that you can use to organize your model and dataset artifacts called **Models** and **Datasets**, respectively. You can create [additional registries to organize other artifact types based on your organization's needs]({{< relref "./registry_types.md" >}}). 

Each [registry]({{< relref "./configure_registry.md" >}}) consists of one or more [collections]({{< relref "./create_collection.md" >}}). Each collection represents a distinct task or use case.

{{< img src="/images/registry/homepage_registry.png" alt="W&B Registry" >}}

To add an artifact to a registry, you first log a [specific artifact version to W&B]({{< relref "/guides/core/artifacts/create-a-new-artifact-version.md" >}}). Each time you log an artifact, W&B automatically assigns a version to that artifact. Artifact versions use 0 indexing, so the first version is `v0`, the second version is `v1`, and so on. 

Once you log an artifact to W&B, you can then link that specific artifact version to a collection in the registry. 

{{% alert %}}
The term "link" refers to pointers that connect where W&B stores the artifact and where the artifact is accessible in the registry. W&B does not duplicate artifacts when you link an artifact to a collection.
{{% /alert %}}

As an example, the proceeding code example shows how to log and link a model artifact called "my_model.txt" to a collection named "first-collection" in the [core registry]({{< relref "./registry_types.md" >}}):

1. Initialize a W&B Run.
2. Log the artifact to W&B.
3. Specify the name of the collection and registry to link your artifact version to.
4. Link the artifact to the collection.

Save this Python code to a script and run it. W&B Python SDK version 0.18.6 or newer is required.

```python title="hello_collection.py"
import wandb
import random

# Initialize a W&B Run to track the artifact
run = wandb.init(project="registry_quickstart") 

# Create a simulated model file so that you can log it
with open("my_model.txt", "w") as f:
   f.write("Model: " + str(random.random()))

# Log the artifact to W&B
logged_artifact = run.log_artifact(
    artifact_or_path="./my_model.txt", 
    name="gemma-finetuned", 
    type="model" # Specifies artifact type
)

# Specify the name of the collection and registry
# you want to publish the artifact to
COLLECTION_NAME = "first-collection"
REGISTRY_NAME = "model"

# Link the artifact to the registry
run.link_artifact(
    artifact=logged_artifact, 
    target_path=f"wandb-registry-{REGISTRY_NAME}/{COLLECTION_NAME}"
)
```

W&B automatically creates a collection for you if the collection you specify in the returned run object's `link_artifact(target_path = "")` method does not exist within the registry you specify.

{{% alert %}}
The URL that your terminal prints directs you to the project where W&B stores your artifact. 
{{% /alert %}}

Navigate to the Registry App to view artifact versions that you and other members of your organization publish. To do so, first navigate to W&B. Select **Registry** in the left sidebar below **Applications**. Select the "Model" registry. Within the registry, you should see the "first-collection" collection with your linked artifact version.

Once you link an artifact version to a collection within a registry, members of your organization can view, download, and manage your artifact versions, create downstream automations, and more if they have the proper permissions.

{{% alert %}}
If an artifact version logs metrics (such as by using `run.log_artifact()`), you can view metrics for that version from its details page, and you can compare metrics across artifact versions from the collection's page. Refer to [View linked artifacts in a registry]({{< relref "link_version.md#view-linked-artifacts-in-a-registry" >}}).
{{% /alert %}}

## Enable W&B Registry

Based on your deployment type, satisfy the following conditions to enable W&B Registry:

| Deployment type | How to enable |
| ----- | ----- |
| Multi-tenant Cloud | No action required. W&B Registry is available on the W&B App. |
| Dedicated Cloud | Contact your account team to enable W&B Registry for your deployment. |
| Self-Managed | Set the environment variable `ENABLE_REGISTRY_UI` to `true`. Refer to [Configure environment variables]({{< relref "/guides/hosting/env-vars.md" >}}). Requires Server v0.59.2 or newer. |


## Resources to get started

Depending on your use case, explore the following resources to get started with the W&B Registry:

* Check out the tutorial video:
    * [Getting started with Registry from W&B](https://www.youtube.com/watch?v=p4XkVOsjIeM)
* Take the W&B [Model CI/CD](https://www.wandb.courses/courses/enterprise-model-management) course and learn how to:
    * Use W&B Registry to manage and version your artifacts, track lineage, and promote models through different lifecycle stages.
    * Automate your model management workflows using webhooks.
    * Integrate the registry with external ML systems and tools for model evaluation, monitoring, and deployment.



## Migrate from the legacy Model Registry to W&B Registry

The legacy Model Registry is scheduled for deprecation with the exact date not yet decided. Before deprecating the legacy Model Registry, W&B will migrate the contents of the legacy Model Registry to the W&B Registry. 


See [Migrating from legacy Model Registry]({{< relref "./model_registry_eol.md" >}}) for more information about the migration process from the legacy Model Registry to W&B Registry.

Until the migration occurs, W&B supports both the legacy Model Registry and the new Registry. 

{{% alert %}}
To view the legacy Model Registry, navigate to the Model Registry in the W&B App. A banner appears at the top of the page that enables you to use the legacy Model Registry App UI.

{{< img src="/images/registry/nav_to_old_model_reg.gif" alt="Legacy Model Registry UI" >}}
{{% /alert %}}


Reach out to support@wandb.com with any questions or to speak to the W&B Product Team about any concerns about the migration.