---
slug: /guides/automations
description: Automate common workflows easily
---

# Automations

Automate and execute your model development workflows right from W&B with Automations.

## Automations features
* **Model CI**: Automatically test new models
* **Deployment**: Easily trigger model deployment
* **Reporting**: Automate report generation for your team

![](/images/models/automations_sidebar_step_1.png)

## How it works
Track and manage your trained models with a few simple steps.

1. **Log model versions**: In your training script, add a couple lines of code to save the model files as an artifact to W&B.
2. **Compare performance**: Check live charts to compare the metrics and sample predictions from model training and validation. Identify which model version performed the best.
3. **Link to registry**: Bookmark the best model version by linking it to a registered model, either programmatically in Python or manually in the W&B UI.
4. **Test and deploy**: Transition model versions through customizable workflows stages, such as `staging` and `production`.

## How to get started
[Try the Quickstart](/guides/models/quickstart) to log and link a sample model in just two minutes.