---
slug: /guides/registry
displayed_sidebar: default
---

# Registry

:::info
W&B Registry is now in public preview. Visit [this](#enable-wb-registry) section to learn how to enable it for your deployment type.
:::

W&B Registry is a curated central repository that stores and provides versioning, aliases, lineage tracking, and governance of models and datasets. Use W&B Registry to share and collaboratively manage the lifecycle of all models, datasets and other artifacts individuals and teams across the entire organization.

<!-- As the single source of truth for which models are in production, Registry provides the foundation for an effective CI/CD pipeline by identifying the right models to reproduce, retrain, evaluate, and deploy. -->

![](/images/registry/registry_landing_page.png)

With W&B Registry you can:

- [Bookmark](./link_version.md) your best artifacts for each machine learning task.
- [Automate](../model_registry/model-registry-automations.md) downstream processes and model CI/CD.
- Track an [artifact’s lineage](../model_registry/model-lineage.md) and audit the history of changes to production artifacts.
- [Configure](./configure_registry.md) viewer, member, or admin access to a registry for all org users.
- Quickly find or reference important artifacts with a unique identifier known as aliases.

## How it works
W&B Registry is composed of three main components: registries, collections, and [artifact versions](../artifacts/create-a-new-artifact-version.md).

A *registry* is a repository or catalog for ML assets of the same kind. You can think of a registry as the top most level of a directory. Each registry consists of one or more sub directories called collections. A *collection* is a folder or a set of linked [artifact versions](../artifacts/create-a-new-artifact-version.md) inside a registry. An [*artifact version*](../artifacts/create-a-new-artifact-version.md) is a single, immutable snapshot of an artifact at a particular stage of its development.  A registry belongs to an organization, not a specific team.

![](/images/registry/registry_diagram_homepage.png)


<!-- Bookmark the most relevant and valuable artifact version by linking it to a registry (see line 18). -->

Track and publish your artifacts to W&B Registry with four major steps:

1. Initialize a W&B run object with [`wandb.init()`](../../ref/python/init.md)
2. Create an artifact with [`wandb.Artifact`](../../ref/python/artifact.md) and add one or more files, directory, or external reference you want to track.
3. Create the target path where you want to link your artifact version to. The path contains the organization entity your team belongs to, the name of the registry, and the name of the collection.
4. Link the artifact to a collection with the run object's [`link_artifact`](../../ref/python/run.md#link_artifact) method. 

```python showLineNumbers
import wandb

org_entity = "<organization_entity>" # Entity of your organization
registry_name = "<registry_name>" # Name of registry that has your collection
collection_name = "<collection_name>" # Name of the collection you want to link to

# Initializes a new W&B run
run = wandb.init(project="<project_name>") 

# Create an artifact object
# Assign it a type such as 'data', 'model' and so forth
artifact = wandb.Artifact(name = "<artifact_name>", type = "<type>")

# Add files to the artifact
artifact.add_file(local_path = "<path_to_file>")
artifact.save() # Save changes to artifact

# The full path of the collection and registry
path = f"{org_entity}/wandb-registry-{registry_name}/{collection_name}"

# Link artifact to the collection in the registry 
run.link_artifact(artifact = artifact, target_path = path)

run.finish()
```



<details>
<summary>Example</summary>

The proceeding code sample logs and link a simulated model file called `my_model.txt` to the Model registry. 

1. First, initialize a run. The run, and the artifacts logged to it, appear in a project called "registry_quickstart". 
2. For demonstrative purposes, simulate logging model metrics that occur during a training run.
3. For demonstrative purposes, create a mock model file.
4. Next, log the simulated model file to the run as an artifact called "gemma-finetuned-3twsov9e". Note that, because we link the artifact to the Model registry (see next step), we specify `"model"` as the artifact's type (`type="model"`).
5. Lastly, link the artifact to a registry called "quickstart-collection" within the Models registry. Ensure to provide the entity of your organization for the `org_entity` variable.

Copy and paste the proceeding code snippet into a Jupyter notebook or Python script:


```python showLineNumbers
import wandb
import random

# Start a new W&B run to track your experiment
run = wandb.init(project="registry_quickstart") 

# Simulate logging model metrics
run.log({"acc": random.random()})

# Create a pseudo model file
with open("my_model.txt", "w") as f:
   f.write("Model: " + str(random.random()))

# log an artifact version 
logged_artifact = run.log_artifact(
    artifact_or_path="./my_model.txt", 
    name="gemma-finetuned-3twsov9e", 
    type="model"
    )

# Provide the name of your organization
org_entity = "<organization_entity>"

# The name of the registry
registry = "model"

# The name of the collection
collection = "quickstart-collection"

# link the model to the predefined core Models registry 
run.link_artifact(
    artifact=logged_artifact, 
    target_path=f"{org_entity}/wandb-registry-{registry}/registry-{collection}"
    )

run.finish()
```

Once the code completes, your notebook or terminal will provide links to the W&B App UI where you can view your project or run.

</details>





## Enable W&B Registry

Based on your deployment type, satisfy the following conditions to enable W&B Registry:

| Deployment type | How to enable |
| ----- | ----- |
| Multi-tenant Cloud | No action required. W&B Registry is available on the W&B App. |
| Dedicated Cloud | Contact your account team. The Solutions Architect (SA) Team will enable W&B Registry with your instance's operator console. Ensure your instance is on server release version 0.57.2 or newer.|
| Self-Managed   | Enable the environment variable called `ENABLE_REGISTRY_UI`. To learn more about enabling environment variables in server, visit [these docs](https://docs.wandb.ai/guides/hosting/env-vars). In self-managed instances, your infrastructure admin should enable this environment variable and set it to `true`. Ensure your instance is on server release version 0.57.2 or newer.|


## Resources to get started

Depending on your use case, explore the following resources to get started with the W&B Registry:

- Check out the tutorial video:
    - [Getting started with Registry from Weights & Biases](https://www.youtube.com/watch?v=p4XkVOsjIeM)

- Take the W&B [Model CI/CD](https://www.wandb.courses/courses/enterprise-model-management) course and learn how to:
    - Use W&B Registry to manage and version your artifacts, track lineage, and promote models through different lifecycle stages.
    - Automate your model management workflows using webhooks and launch jobs.
    - See how Registry integrates with external ML systems and tools in your model development lifecycle for model evaluation, monitoring, and deployment.

## Migrating from the legacy Model Registry to W&B Registry

The W&B Model Registry will be deprecated by the end of 2024. The contents in your Model Registry will be migrated to W&B Registry. For more information about the migration process, see [Migrating from legacy Model Registry to Registry](./model_registry_eol.md). 

The legacy W&B Model Registry App UI is still available until W&B Registry is available to all users. To view the legacy Model Registry: Navigate to the Model Registry from the homepage. A banner will appear to view the legacy Model Registry App UI.

![](/images/registry/nav_to_old_model_reg.gif)

Reach out to support@wandb.com with any questions or to speak to our product team about any concerns with the migration.


