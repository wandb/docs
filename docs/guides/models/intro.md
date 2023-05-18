---
slug: /guides/models
description: Model registry to manage the model lifecycle from training to production
---

# Models 

Use W&B Models as a central system of record for your best models, standardized and organized in a model registry across projects and teams.

## Model registry features
* **Versioning**: Bookmark your best model versions for each machine learning task
* **Lifecycle**: Move model versions through the lifecycle from staging to production
* **Lineage**: Audit the history of changes to production models

![](/images/models/models_landing_page.png)

## How it works
Track and manage your trained models with a few simple steps.

1. **Log model versions**: In your training script, add a couple lines of code to save the model files as an artifact to W&B.
2. **Compare performance**: Check live charts to compare the metrics and sample predictions from model training and validation. Identify which model version performed the best.
3. **Link to registry**: Bookmark the best model version by linking it to a registered model, either programmatically in Python or manually in the W&B UI.
4. **Test and deploy**: Transition model versions through customizable workflows stages, such as `staging` and `production`.

## How to get started
[Try the Quickstart](./quickstart.md) to log and link a sample model in just two minutes.



