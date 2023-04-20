---
slug: /guides/models
description: Manage the model lifecycle from training to production
---

# Model Registry

Use Model Registry as the central system to organize your models and their versions for a given task. Easily keep track and promote models in their various stages of maturity or task: from staging, to production, and more. View the change history of registered model on the W&B App.

![](/images/models/model_registry_landing_page.png)

The image above shows the Model Registry W&B App UI. The left panel demonstrates a list of registered models. On the right panel there is a **Model Overview** that describes INSERT. On the bottom right of the image is the **Versions** section. This section lists all the models versions created, when they were created, aliases associated with a specific version, and more.


<!-- ### Watch the 1 minute video walkthrough -->

<!-- {% embed url="https://www.youtube.com/watch?v=jy9Pk9riwZI" %} -->

## How it works
There are three major components to Model Registry: model versions, model artifacts, and registered models.

* Model versions: a package of data & metadata describing a trained model.
* Model artifact: a sequence of logged model versions.
* Registered models: a selection of linked model versions. Registered models often represent all of the candidate models for a single modeling use case or task.



## How to get started
* Read [Model Management Concepts](./model-management-concepts.md) for more information on basic Model Management concepts.
* Follow the steps in the [Walkthrough](./walkthrough.md) to learn how to use Model Registry.


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
* 10 most recent steps of Action History shown in the UI for each registered model -->




