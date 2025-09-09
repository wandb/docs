---
aliases:
- /guides/core/registry/model_registry
cascade:
- url: /guides/registry/model_registry/:filename
description: Model registry to manage the model lifecycle from training to production
menu:
  default:
    identifier: ja-guides-core-registry-model_registry-_index
    parent: registry
title: Model registry
url: /guides/registry/model_registry
weight: 9
---

{{% alert %}}
W&B will eventually stop supporting W&B Model Registry. Users are encouraged to instead use [W&B Registry]({{< relref path="/guides/core/registry/" lang="ja" >}}) for linking and sharing their model artifacts versions. W&B Registry broadens the capabilities of the legacy W&B Model Registry. For more information about W&B Registry, see the [Registry docs]({{< relref path="/guides/core/registry/" lang="ja" >}}).


W&B will migrate existing model artifacts linked to the legacy Model Registry to the new W&B Registry in the near future. See [Migrating from legacy Model Registry]({{< relref path="/guides/core/registry/model_registry_eol.md" lang="ja" >}}) for information about the migration process.
{{% /alert %}}

The W&B Model Registry houses a team's trained models where ML Practitioners can publish candidates for production to be consumed by downstream teams and stakeholders. It is used to house staged/candidate models and manage workflows associated with staging.

{{< img src="/images/models/model_reg_landing_page.png" alt="Model Registry" >}}

With W&B Model Registry, you can:

* [Bookmark your best model versions for each machine learning task.]({{< relref path="./link-model-version.md" lang="ja" >}})
* [Automate]({{< relref path="/guides/core/automations/" lang="ja" >}}) downstream processes and model CI/CD.
* Move model versions through its ML lifecycle; from staging to production.
* Track a model's lineage and audit the history of changes to production models.

{{< img src="/images/models/models_landing_page.png" alt="Models overview" >}}

## How it works
Track and manage your staged models with a few simple steps.

1. **Log a model version**: In your training script, add a few lines of code to save the model files as an artifact to W&B. 
2. **Compare performance**: Check live charts to compare the metrics and sample predictions from model training and validation. Identify which model version performed the best.
3. **Link to registry**: Bookmark the best model version by linking it to a registered model, either programmatically in Python or interactively in the W&B UI.

The following code snippet demonstrates how to log and link a model to the Model Registry:

```python
import wandb
import random

# Start a new W&B run
run = wandb.init(project="models_quickstart")

# Simulate logging model metrics
run.log({"acc": random.random()})

# Create a simulated model file
with open("my_model.h5", "w") as f:
    f.write("Model: " + str(random.random()))

# Log and link the model to the Model Registry
run.link_model(path="./my_model.h5", registered_model_name="MNIST")

run.finish()
```

4. **Connect model transitions to CI/CD workflows**: transition candidate models through workflow stages and [automate downstream actions]({{< relref path="/guides/core/automations/" lang="ja" >}}) with webhooks.


## How to get started
Depending on your use case, explore the following resources to get started with W&B Models:

* Check out the two-part video series:
  1. [Logging and registering models](https://www.youtube.com/watch?si=MV7nc6v-pYwDyS-3&v=ZYipBwBeSKE&feature=youtu.be)
  2. [Consuming models and automating downstream processes](https://www.youtube.com/watch?v=8PFCrDSeHzw) in the Model Registry.
* Read the [models walkthrough]({{< relref path="./walkthrough.md" lang="ja" >}}) for a step-by-step outline of the W&B Python SDK commands you could use to create, track, and use a dataset artifact.
* Learn about:
   * [Protected models and access control]({{< relref path="./access_controls.md" lang="ja" >}}).
   * [How to connect Registry to CI/CD processes]({{< relref path="/guides/core/automations/" lang="ja" >}}).
   * Set up [Slack notifications]({{< relref path="./notifications.md" lang="ja" >}}) when a new model version is linked to a registered model.
* Review [What is an ML Model Registry?](https://wandb.ai/wandb_fc/model-registry-reports/reports/What-is-an-ML-Model-Registry---Vmlldzo1MTE5MjYx) to learn how to integrate Model Registry into your ML workflow. 
* Take the W&B [Enterprise Model Management](https://www.wandb.courses/courses/enterprise-model-management) course and learn how to:
  * Use the W&B Model Registry to manage and version your models, track lineage, and promote models through different lifecycle stages
  * Automate your model management workflows using webhooks.
  * See how the Model Registry integrates with external ML systems and tools in your model development lifecycle for model evaluation, monitoring, and deployment.