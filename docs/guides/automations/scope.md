---
description: Determine the scope of your automations.
displayed_sidebar: default
---

# Automation scope
Each automation has a different scope, which determines what an automation applies to. Currently, there are two scope types: Registry Automation and Project Automations.

## Automations in a Registry
Registry-scoped automations apply to a single collection in a registry. You can create automations for each registry you have.

### Example use cases

| Automation Process | Artifact Type | Event Type |Potential Action |
| --- | --- | --- | --- |
| Prepare and package a model version for deployment | A Collection of type `model` inside a registry | Adding a specific alias to a model version | Model Packaging, Serialization, Optimization |
| Model Evaluation | A Collection of type `model` inside a registry | Adding a specific alias to a model version | Trigger the creation of a W&B Report with evaluation results |

## Automations in a Project
Project-scoped automations apply to either all collections in a project or a single collection in a project.

### Example use cases 

| Automation Process | Artifact Type | Event Type |Potential Action |
| --- | --- | --- | --- |
| Hyperparameter Sweep | An Artifact Collection of type `train-dataset` inside a project | Add a new artifact version (of type `train-dataset`) | Trigger a hyperparameter sweep for a model trained on a new dataset |
| Retrain a Model when new data is uploaded | An Artifact Collection of type `train-dataset` inside a project | Add a new artifact version (of type `train-dataset`) | Trigger a model training job every time training data is refreshed |
| Data drift analysis | An Artifact Collection of type `train-dataset` inside a project | Add a new artifact version (of type `train-dataset`) | Run a dataset drift analysis script when a new dataset is uploaded to compare against a baseline dataset |
