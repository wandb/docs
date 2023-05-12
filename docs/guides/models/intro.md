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
[Try the Quickstart](/guides/models/quickstart) to log and link a sample model in just two minutes.

<!-- 
Use the W&B Model Registry as a central system of record for models.

* Create Registered Models to organize your best model versions for a given task
* Track a model moving into staging and production
* See a history of all changes, including who moved a model to production -->

<!-- ### Watch the 1 minute video walkthrough -->

<!-- {% embed url="https://www.youtube.com/watch?v=jy9Pk9riwZI" %} -->

<!-- ## Model Registry Features

### Model Versioning

Iterate to get the best model version for a task, and catalog all the changes along the way.

* Track every model version in a central repository
* Browse and compare model versions
* Capture training metrics and hyperparameters

### Model Lineage

Document and reproduce the complete pipeline of model training and evaluation.

* Identify the exact dataset version the model trained on
* Restore the training code, including git commit and diff patch
* Get back to the model’s hyperparameters and other metadata for reproducibility
* Dig in to upstream jobs that can affect model performance

### Model Lifecycle

Manage the process as a model moves from training through staging to production.

* Highlight the best model versions that are being evaluated for production
* Communicate where a model version is in the process — staging, production etc
* Review the history of model versions that moved through each stage

## Model Registry Pilot Limits

This new feature is now turned on for all users to try for free, up to:

* 5 Registered Models, with unlimited versions linked to each model
* 10 most recent steps of Action History shown in the UI for each registered model

## Explore more
* Read [Model Management Concepts](./model-management-concepts.md) for more information on basic Model Management concepts.
* Follow the steps in the [Walkthrough](./walkthrough.md) to learn how to use Model Management. -->

