---
slug: /guides/registry
displayed_sidebar: default
---

# Registry

W&B Registry is a curated and governed repository of machine learning [artifacts](../artifacts/intro.md) within your W&B organization. The W&B Registry provides artifact versioning, artifact lineage tracking, provides information of when an artifact is created and when an artifact is used, and more.

![](/images/registry/registry_landing_page.png)

Use W&B Registry to:

- [Bookmark](./link_version.md) your best artifacts for each machine learning task.
- [Automate](../model_registry/model-registry-automations.md) downstream processes and model CI/CD.
- Track an [artifact’s lineage](../model_registry/model-lineage.md) and audit the history of changes to production artifacts.
- [Configure](./configure_registry.md) viewer, member, or admin access to a registry for all org users
- Quickly find or reference important artifacts with a unique identifier known as aliases.

## How it works

Track and publish your staged artifacts to W&B Registry in a few steps:

1. Log an artifact version: In your training or experiment script, add a few lines of code to save the artifact to a W&B run.
2. Link to registry: Bookmark the most relevant and valuable artifact version by linking it to a registry.

The following code snippet demonstrates how to log and link a model to the model registry inside W&B Registry:

```python
import wandb
import random

# Start a new W&B run to track your experiment
run = wandb.init(project="registry_quickstart") 

# Simulate logging model metrics
run.log({"acc": random.random()})

# Create a simulated model file
with open("my_model.txt", "w") as f:
   f.write("Model: " + str(random.random()))

# log and link the model to the model registry inside W&B Registry
logged_artifact = run.log_artifact(artifact_or_path="./my_model.txt", name="gemma-finetuned-3twsov9e", type="model")
run.link_artifact(artifact=logged_artifact, target_path=f"<INSERT-ORG-NAME>/wandb-registry-model/registry-quickstart-collection"),

run.finish()
```
See learn more about linking to a registry, visit [this](https://docs.wandb.ai/guides/registry/link_version) guide. 

## How to access Registry

Registry is in public preview. The table below shows how to enable it depending on your deployment type:

| Deployment Type    | Access Method |
|--------------------|---------------|
| **Public Cloud**   | By default, already available in the UI. |
| **Self-Managed**   | Enable the environment variable called `ENABLE_REGISTRY_UI`. To learn more about enabling environment variables in server, visit [these docs](https://docs.wandb.ai/guides/hosting/env-vars). In self-managed instances, the customer's infrastructure admin should enable this environment variable and set it to `true`. |
| **Dedicated Cloud** | Contact your account team. The Solutions Architect (SA) team will enable it via the instance's operator console. |

## Migrating from the W&B Model Registry to the W&B Registry

If your team is actively using the existing W&B Model Registry to organize your models, this will still be available through the new Registry App UI. Navigate to the Model Registry from the homepage, and the banner will allow to select a team and visit it's model registry.

![](/images/registry/nav_to_old_model_reg.gif)

Look out for incoming information on a migration we will be making available to migrate contents from the current model registry into the new model registry inside W&B Registry. You can reach out to support@wandb.com with any questions or to speak to our product team about any concerns with the migration.

 
