---
slug: /guides/registry
displayed_sidebar: default
---

# Registry

:::info
W&B Registry is in private preview. Contact your account team or support@wandb.com for early access.  
:::

W&B Registry is a curated and governed repository of machine learning [artifacts](../artifacts/intro.md) within your W&B organization. The W&B Registry provides artifact versioning, artifact lineage tracking, provides information of when an artifact is created and when an artifact is used, and more.

![](/images/registry/registry_landing_page.png)

Use W&B Registry to:

- [Bookmark](./link_version.md) your best artifacts for each machine learning task.
- [Automate](../model_registry/model-registry-automations.md) downstream processes and model CI/CD.
- Track an [artifact’s lineage](../model_registry/model-lineage.md) and audit the history of changes to production artifacts.
- [Configure](./configure_registry.md) viewer, member, or admin access to a registry for all org users
- Quickly find or reference important artifacts with a unique identifier known as aliases.

## How it works
W&B Registry is composed of three main components: registries, collections, and [artifact versions](../artifacts/create-a-new-artifact-version.md).

A *registry* is a repository or catalog for ML assets of the same kind. You can think of a registry as the top most level of a directory. Each registry consists of one or more sub directories called collections. A *collection* is a folder or a set of linked [artifact versions](../artifacts/create-a-new-artifact-version.md) inside a registry. An [*artifact version*](../artifacts/create-a-new-artifact-version.md) is a single, immutable snapshot of an artifact at a particular stage of its development.  A registry belongs to an organization, not a specific team.

![](/images/registry/registry_diagram_homepage.png)


<!-- Bookmark the most relevant and valuable artifact version by linking it to a registry (see line 18). -->

Track and publish your artifacts to W&B Registry with three major steps:

1. Initialize a W&B run object with [`wandb.init()`](../../ref/python/init.md)
2. Log an artifact version to the run using the run object's [`log_artifact`](../../ref/python/run.md#log_artifact) method.
3. Link the artifact to a collection with the run object's [`link_artifact`](../../ref/python/run.md#link_artifact) method. 

```python showLineNumbers
import wandb

# Start a new W&B run to track your experiment
run = wandb.init(project="<project_name>") 

# log an artifact version 
logged_artifact = run.log_artifact(
    artifact_or_path="<artifact>", 
    name="<artifact_name>", 
    type="<type>"
    )

# Provide the entity of your organization
org_entity = "<organization_entity>"

# The name of the registry you want to link to
registry_name = "<registry_name>"

# The name of the collection you want to link to. 
# If the collection does not exist, W&B creates one for you
collection_name = "<collection_name>"

# The full path of the collection and registry
path = f"{org_entity}/wandb-registry-{registry_name}/{collection_name}"

# Link artifact to the collection in the registry 
# specified in target_path
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





## How to get started

Depending on your use case, explore the following resources to get started with the W&B Registry:

* [Create a custom registry](./create_registry.md).
* Once you have a custom registry, learn how [to create your first collection](./create_collection.md).
* Learn how to [link an artifact to a collection](./link_version.md).
* [Configuring access control](./configure_registry.md) for a registry
* Learn how to [connect the Models registry to CI/CD processes](../model_registry/model-registry-automations.md).

For detailed information on how to integrate W&B Registry to your ML workflow, consider the following resources:

- Check out the two-part video series on the model registry:
    - [Logging and registering models](https://www.youtube.com/watch?si=MV7nc6v-pYwDyS-3&v=ZYipBwBeSKE&feature=youtu.be)
    - [Consuming artifacts and automating downstream processes](https://www.youtube.com/watch?v=8PFCrDSeHzw) in Registry.
- Take the W&B [Model CI/CD](https://www.wandb.courses/courses/enterprise-model-management) course and learn how to:
    - Use the W&B Registry to manage and version your artifacts, track lineage, and promote models through different lifecycle stages
    - Automate your model management workflows using webhooks and launch jobs.
    - See how Registry integrates with external ML systems and tools in your model development lifecycle for model evaluation, monitoring, and deployment.

## Migrating from the W&B Model Registry to the W&B Registry

If your team is actively using the existing W&B Model Registry to organize your models, this will still be available through the new Registry App UI. Navigate to the Model Registry from the homepage, and the banner will allow to select a team and visit it's model registry.

![](/images/registry/nav_to_old_model_reg.gif)

Look out for incoming information on a migration we will be making available to migrate contents from the current model registry into the new model registry inside W&B Registry. You can reach out to support@wandb.com with any questions or to speak to our product team about any concerns with the migration.

 
