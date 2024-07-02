---
slug: /guides/registry
displayed_sidebar: default
---

# Registry

:::info
W&B Registry is in private preview. Contact your account team or support@wandb.com for early access.  
:::

W&B Registry is a governed and curated central repository of machine learning artifacts, and provides versioning, aliasing, and lineage tracking of all ML models, datasets, and other artifacts across your organization. Registry provides ML practitioners the ability to track the creation and usage of artifacts related to ML experiments.

![Registry-landing-page-homepage-with-custom-registries](https://github.com/wandb/docodile/assets/40642416/51a929c9-82af-4015-8183-82ea4de32026)

With W&B Registry, you can:

- [Bookmark](https://docs.wandb.ai/guides/registry/link_version) your best artifacts for each machine learning task.
- [Automate](https://docs.wandb.ai/guides/model_registry/model-registry-automations) downstream processes and model CI/CD.
- Track an [artifact’s lineage](https://docs.wandb.ai/guides/model_registry/model-lineage) and audit the history of changes to production artifacts.
- [Configure](https://docs.wandb.ai/guides/registry/configure_registry) viewer, member, or admin access to a registry for all org users

## How it works

Track and publish your staged artifacts to W&B Registry in a few steps.

1. Log an artifact version: In your training or experiment script, add a few lines of code to save the artifact to a W&B run.
2. Link to registry: Bookmark the most relevant and valuable artifact version by linking it to a registry.

The following code snippet demonstrates how to log and link a model to the model registry inside W&B Registry:

```jsx
import wandb
import random

# Start a new W&B run to track your experiment
run = wandb.init(project="registry_quickstart") 

# Simulate logging model metrics
run.log({"acc": random.random()})

# Create a simulated model file
with open("my_model.h5", "w") as f:
   f.write("Model: " + str(random.random()))

# log and link the model to the model registry inside W&B Registry
logged_artifact = run.log_artifact(artifact_or_path="./my_model.h5", name="gemma-finetuned-3twsov9e", type="model")
run.link_artifact(artifact=logged_artifact, target_path="<YOUR-ORG-NAME>/wandb-registry-model/registry-quickstart-collection"),

run.finish()
```

1. When working with models, connect transitions to CI/CD workflows: transition candidate models through workflow stages and [automate downstream actions](https://docs.wandb.ai/guides/model_registry/model-registry-automations) with webhooks or jobs.

## How to get started

Depending on your use case, explore the following resources to get started with the W&B Registry:

- Check out the two-part video series on the model registry:
    - [Logging and registering models](https://www.youtube.com/watch?si=MV7nc6v-pYwDyS-3&v=ZYipBwBeSKE&feature=youtu.be)
    - [Consuming artifacts and automating downstream processes](https://www.youtube.com/watch?v=8PFCrDSeHzw) in Registry.
- Learn about:
    - [Configuring access control](https://docs.wandb.ai/guides/registry/configure_registry) for a registry
    - [How to connect the Model Registry to CI/CD processes](https://docs.wandb.ai/guides/model_registry/model-registry-automations).
- Take the W&B [Model CI/CD](https://www.wandb.courses/courses/enterprise-model-management) course and learn how to:
    - Use the W&B Registry to manage and version your artifacts, track lineage, and promote models through different lifecycle stages
    - Automate your model management workflows using webhooks and launch jobs.
    - See how Registry integrates with external ML systems and tools in your model development lifecycle for model evaluation, monitoring, and deployment.

## Migrating from the W&B Model Registry to the W&B Registry

If your team is actively using the existing W&B Model Registry to organize your models, this will still be available through the new Registry App UI. Navigate to the Model Registry from the homepage, and the banner will allow to select a team and visit it's model registry.

![](/images/registry/nav_to_old_model_reg.gif)

Look out for incoming information on a migration we will be making available to migrate contents from the current model registry into the new model registry inside W&B Registry. You can reach out to support@wandb.com with any questions or to speak to our product team about any concerns with the migration.

 

<!-- To do: INSERT -->
